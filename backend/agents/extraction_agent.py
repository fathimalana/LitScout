import concurrent.futures
import os
import json
import re
import time
import random
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import difflib
from typing import List, Dict, TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# PDF processing imports
import fitz  # PyMuPDF

load_dotenv()

# ========================
# HELPER FUNCTIONS
# ========================

def safe_print(text):
    """Print text safely, handling Unicode encoding errors"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII, replacing problematic characters
        print(text.encode('ascii', 'replace').decode('ascii'))

# Phase 1 Improvements: User-agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def get_random_user_agent() -> str:
    """Get a random user agent to avoid blocking"""
    return random.choice(USER_AGENTS)


class ExtractionState(TypedDict):
    screened_papers: List[Dict]
    papers_with_pdfs: List[Dict]
    papers_with_text: List[Dict]
    extraction_results: Dict



def fetch_pdf_from_arxiv(arxiv_id: str) -> Optional[bytes]:
    """
    Fetch PDF directly from arXiv using arXiv ID
    """
    if not arxiv_id:
        return None
    
    try:
        # Use export subdomain as per arXiv best practices
        pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
        
        headers = {'User-Agent': 'LitScoutResearchBot/1.0'}
        
        print(f"  Fetching arXiv PDF: {arxiv_id}")
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.get(pdf_url, headers=headers, timeout=45)
            if response.status_code == 429:
                sleep_time = (attempt + 1) * 3
                print(f"  [RATE LIMIT] arXiv PDF 429. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            break
        else:
            return None
        
        return response.content
        
    except Exception as e:
        print(f"  Error fetching arXiv PDF: {e}")
        return None


def fetch_pdf_from_open_access(open_access_info: Dict) -> Optional[bytes]:
    """
    Fetch PDF from OpenAccess URL if available
    """
    if not open_access_info or not isinstance(open_access_info, dict):
        return None
    
    pdf_url = open_access_info.get("url", "")
    
    # Check if URL is valid and not empty
    if not pdf_url or pdf_url == "" or len(pdf_url) < 10:
        return None
    
    try:
        print(f"  Fetching OpenAccess PDF from: {pdf_url[:50]}...")
        
        headers = {'User-Agent': get_random_user_agent(), 'Accept': 'application/pdf,*/*'}
        response = requests.get(pdf_url, headers=headers, timeout=45)
        response.raise_for_status()
        
        # Verify it's actually a PDF
        if response.content[:4] == b'%PDF':
            return response.content
        else:
            print(f"  [SKIP] URL did not return a PDF")
            return None
        
    except Exception as e:
        print(f"  Error fetching OpenAccess PDF: {e}")
        return None


def search_arxiv_for_pdf(title: str) -> Optional[bytes]:
    """
    Search arXiv API by title to find PDF
    Improved with better similarity matching
    """
    if not title:
        return None
        
    try:
        # Clean title for query
        clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
        words = [w for w in clean_title.split() if len(w) > 2]
        if not words:
            return None
        
        # Build query like: ti:word1 AND ti:word2 AND ti:word3
        # Limit to 6 words to avoid overly restrictive queries or token limits
        query = "+AND+".join(f"ti:{w}" for w in words[:8])
        
        # We don't need to quote the operators if we're directly injecting +AND+, but let's be safe
        encoded_title = query
        
        api_url = f"http://export.arxiv.org/api/query?search_query={encoded_title}&start=0&max_results=3"
        
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.get(api_url, timeout=10)
            if response.status_code == 429:
                sleep_time = (attempt + 1) * 3
                print(f"  [RATE LIMIT] arXiv API 429. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            break
        else:
            print("  [FAIL] Max retries reached for arXiv API (429)")
            return None
        
        # Parse Atom XML
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        if not entries:
            return None
        
        # Normalize original title
        norm_title_orig = re.sub(r'\s+', ' ', title.lower())
        
        best_match = None
        best_similarity = 0
        
        # Check all results for best match
        for entry in entries:
            entry_title = entry.find('atom:title', ns).text.strip()
            norm_title_found = re.sub(r'\s+', ' ', entry_title.lower())
            
            similarity = difflib.SequenceMatcher(None, norm_title_orig, norm_title_found).ratio()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
      
        if best_similarity < 0.7:
            print(f"  [SKIP] Best arXiv match too loose ({best_similarity:.2f})")
            return None
        
        # Find PDF link
        pdf_link = None
        for link in best_match.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_link = link.get('href')
                break
        
        if pdf_link:
            print(f"  Found matching arXiv PDF (similarity: {best_similarity:.2f})")
            pdf_response = requests.get(pdf_link, headers={'User-Agent': 'LitScoutResearchBot/1.0'}, timeout=30)
            pdf_response.raise_for_status()
            return pdf_response.content
            
        return None
        
    except Exception as e:
        print(f"  Error searching arXiv: {e}")
        return None


def extract_arxiv_id_from_paper(paper: Dict) -> Optional[str]:
    """
    Try to extract arXiv ID from paper metadata
    Checks: externalIds, abstract, title
    """
    # Check externalIds field
    external_ids = paper.get('externalIds', {})
    if isinstance(external_ids, dict):
        arxiv_id = external_ids.get('ArXiv') or external_ids.get('arxiv')
        if arxiv_id:
            return arxiv_id
    
    # Check abstract for arXiv ID
    abstract = paper.get('abstract', '')
    if abstract:
        match = re.search(r'arXiv:(\d{4}\.\d{4,5})', abstract)
        if match:
            return match.group(1)
    
    # Check URL for arXiv pattern
    url = paper.get('url', '')
    if 'arxiv.org' in url:
        match = re.search(r'(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})', url)
        if match:
            return match.group(1)
    
    return None


# ========================
# TEXT EXTRACTION FUNCTIONS
# ========================

def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip common headers/footers
        if re.match(r'^(page|\d+|figure|table|www\.|http|doi:|arxiv:)', line, re.IGNORECASE):
            continue
        
        # Skip single characters or pure numbers
        if len(line) <= 2 or line.isdigit():
            continue
        
        # Skip copyright/license notices
        if re.search(r'(copyright|©|\(c\)|license|permission|reprinted)', line, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    # Join and normalize whitespace
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text


def extract_text_from_pdf(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Extract text from PDF with IMPROVED section detection
    Handles numbered sections, various formats, and provides fallback logic
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = []
        sections = {
            "title": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "full_text": ""
        }
        
        current_section = None
        
        # Improved section patterns (more flexible)
        # Note: Abstract is NOT extracted as it's already available from Semantic Scholar
        section_patterns = {
            "introduction": [
                r'^\s*introduction\s*:?\s*$',
                r'^\s*\d+\.?\s*introduction',
                r'^\s*[ivxIVX]+\.?\s*introduction',
            ],
            "methods": [
                r'^\s*\d+\.?\s*(method|methodology|approach|materials?)',
                r'^\s*[ivxIVX]+\.?\s*(method|methodology)',
                r'^\s*(method|methodology)\s*:?\s*$',
                r'materials?\s+and\s+methods?',
                r'experimental\s+(setup|design|procedure|methods?)',
            ],
            "results": [
                r'^\s*\d+\.?\s*(result|finding|experiment)s?',
                r'^\s*[ivxIVX]+\.?\s*(result|finding)s?',
                r'^\s*(result|finding)s?\s*:?\s*$',
                r'experimental\s+results?',
            ],
            "discussion": [
                r'^\s*\d+\.?\s*discussion',
                r'^\s*[ivxIVX]+\.?\s*discussion',
                r'^\s*discussion\s*:?\s*$',
                r'results?\s+and\s+discussion',
            ],
            "conclusion": [
                r'^\s*\d+\.?\s*(conclusion|summary|future\s+work)',
                r'^\s*[ivxIVX]+\.?\s*(conclusion|summary)',
                r'^\s*(conclusion|summary)\s*:?\s*$',
            ]
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            for block in blocks:
                text = block[4].strip()
                
                if not text:
                    continue
                
                text_lower = text.lower()
                
                # Title detection (first page only)
                if page_num == 0 and len(sections["title"]) == 0:
                    if len(text.split()) >= 3 and len(text) < 200:
                        sections["title"] = text
                        continue
                
                # Check for references (stop processing)
                if re.search(r'^\s*(reference|bibliography)', text_lower):
                    break
                
                # Section detection (try all patterns)
                section_detected = False
                for section_name, patterns in section_patterns.items():
                    if any(re.search(p, text_lower) for p in patterns):
                        current_section = section_name
                        section_detected = True
                        print(f"  [SECTION] {section_name.upper()}: '{text[:50]}'")
                        break
                
                if section_detected:
                    continue
                
                # Append to current section ONLY if a section has been detected
                # Do NOT use fallback logic that dumps everything into introduction
                if current_section and current_section in sections:
                    sections[current_section] += " " + text
                
                # Always append to full text
                full_text.append(text)
        
        doc.close()
        
        # Clean all sections
        for key in sections:
            sections[key] = clean_pdf_text(sections[key])
        
        sections["full_text"] = ' '.join(full_text)
        sections["full_text"] = clean_pdf_text(sections["full_text"])
        
        # Debug: Print section lengths
        print(f"  [EXTRACTION SUMMARY]")
        for section in ["introduction", "methods", "results", "discussion", "conclusion"]:
            length = len(sections[section].split())
            print(f"    {section}: {length} words")
        
        return sections
        
    except Exception as e:
        print(f"  Error extracting PDF text: {e}")
        return {
            "title": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "full_text": ""
        }


# ========================
# NODE 1: FETCH PDFs
# ========================

def fetch_pdfs_node(state: ExtractionState) -> Dict:
    """
    Fetch PDFs with improved strategy:
    1. Check for arXiv ID in metadata
    2. Try OpenAccess URL if valid
    3. Search arXiv by title with better matching
    """
    print("\n" + "="*80)
    print("EXTRACTION STAGE 1: FETCHING PDFs")
    print("="*80)
    
    screened_papers = state["screened_papers"]
    
    print(f"Processing {len(screened_papers)} papers...")
    
    papers_with_pdfs = []
    fetch_stats = {
        "total_papers": len(screened_papers),
        "arxiv_direct": 0,
        "arxiv_search": 0,
        "open_access": 0,
        "failed": 0,
        "failure_reasons": []
    }
    

    def _fetch_single_paper(paper_info):
        i, paper = paper_info
        safe_print(f"\n[{i}/{len(screened_papers)}] {paper.get('title', 'Untitled')[:60]}...")
        
        pdf_bytes = None
        source = None
        failure_reason = None
        
        # Strategy 1: Check for arXiv ID in metadata
        arxiv_id = extract_arxiv_id_from_paper(paper)
        if arxiv_id:
            pdf_bytes = fetch_pdf_from_arxiv(arxiv_id)
            if pdf_bytes:
                source = "arxiv_direct"
        
        # Strategy 2: Try OpenAccess URL
        if not pdf_bytes:
            open_access_info = paper.get("openAccessPdf")
            pdf_bytes = fetch_pdf_from_open_access(open_access_info)
            if pdf_bytes:
                source = "open_access"
            elif open_access_info and open_access_info.get("url"):
                failure_reason = "OpenAccess URL invalid or not a PDF"
        
        # Strategy 3: Search arXiv by title
        if not pdf_bytes:
            pdf_bytes = search_arxiv_for_pdf(paper.get("title"))
            if pdf_bytes:
                source = "arxiv_search"
            else:
                failure_reason = failure_reason or "No arXiv match found by title"
        
        if pdf_bytes:
            print(f"  [OK] PDF fetched from {source} ({len(pdf_bytes)/1024:.1f} KB)")
            return {
                "success": True,
                "source": source,
                "paper": {
                    **paper,
                    "pdf_bytes": pdf_bytes,
                    "pdf_source": source
                }
            }
        else:
            print(f"  [FAIL] PDF not available - {failure_reason or 'Unknown reason'}")
            return {
                "success": False,
                "title": paper.get("title", "")[:60],
                "reason": failure_reason or "Unknown"
            }

    # Use ThreadPoolExecutor for concurrent fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        indexed_papers = list(enumerate(screened_papers, 1))
        # Map ensures results are in the same relative order
        results = list(executor.map(_fetch_single_paper, indexed_papers))
        
    for res in results:
        if res["success"]:
            papers_with_pdfs.append(res["paper"])
            fetch_stats[res["source"]] += 1
        else:
            fetch_stats["failed"] += 1
            fetch_stats["failure_reasons"].append({
                "title": res["title"],
                "reason": res["reason"]
            })

    
    print(f"\n{'='*80}")
    print(f"PDF Fetch Summary:")
    print(f"  Total papers: {fetch_stats['total_papers']}")
    print(f"  arXiv (direct): {fetch_stats['arxiv_direct']}")
    print(f"  arXiv (search): {fetch_stats['arxiv_search']}")
    print(f"  OpenAccess: {fetch_stats['open_access']}")
    print(f"  Failed: {fetch_stats['failed']}")
    total_success = fetch_stats['arxiv_direct'] + fetch_stats['arxiv_search'] + fetch_stats['open_access']
    print(f"  Success rate: {(total_success / fetch_stats['total_papers'] * 100):.1f}%")
    print(f"{'='*80}\n")
    
    return {
        "papers_with_pdfs": papers_with_pdfs,
        "extraction_results": {"fetch_stats": fetch_stats}
    }


# ========================
# NODE 2: EXTRACT TEXT
# ========================

def extract_text_node(state: ExtractionState) -> Dict:
    """
    Extract text from PDFs with section detection
    """
    print("\n" + "="*80)
    print("EXTRACTION STAGE 2: EXTRACTING TEXT FROM PDFs")
    print("="*80)
    
    papers_with_pdfs = state["papers_with_pdfs"]
    
    print(f"Processing {len(papers_with_pdfs)} PDFs...")
    
    papers_with_text = []
    extraction_stats = {
        "total_pdfs": len(papers_with_pdfs),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "avg_text_length": 0
    }
    
    total_length = 0
    

    def _extract_single_paper(paper_info):
        i, paper = paper_info
        safe_print(f"\n[{i}/{len(papers_with_pdfs)}] Extracting: {paper.get('title', 'Untitled')[:60]}...")
        
        pdf_bytes = paper.get("pdf_bytes")
        
        if not pdf_bytes:
            return {"success": False}
        
        sections = extract_text_from_pdf(pdf_bytes)
        
        if sections["full_text"] and len(sections["full_text"]) > 500:
            # Don't include pdf_bytes in output (too large)
            paper_data = {k: v for k, v in paper.items() if k != "pdf_bytes"}
            
            print(f"  [OK] Extracted {len(sections['full_text'])} chars")
            print(f"    - Title: {'[OK]' if sections['title'] else '[FAIL]'}")
            print(f"    - Introduction: {'[OK]' if sections['introduction'] else '[FAIL]'}")
            print(f"    - Methods: {'[OK]' if sections['methods'] else '[FAIL]'}")
            print(f"    - Results: {'[OK]' if sections['results'] else '[FAIL]'}")
            print(f"    - Conclusion: {'[OK]' if sections['conclusion'] else '[FAIL]'}")
            
            return {
                "success": True,
                "length": len(sections["full_text"]),
                "paper": {
                    **paper_data,
                    "extracted_text": sections
                }
            }
        else:
            print(f"  [FAIL] Extraction failed or text too short")
            return {"success": False}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4)) as executor:
        indexed_pdfs = list(enumerate(papers_with_pdfs, 1))
        results = list(executor.map(_extract_single_paper, indexed_pdfs))
        
    for res in results:
        if res["success"]:
            papers_with_text.append(res["paper"])
            extraction_stats["successful_extractions"] += 1
            total_length += res["length"]
        else:
            extraction_stats["failed_extractions"] += 1
    
    if extraction_stats["successful_extractions"] > 0:
        extraction_stats["avg_text_length"] = total_length / extraction_stats["successful_extractions"]
    
    print(f"\n{'='*80}")
    print(f"Text Extraction Summary:")
    print(f"  Total PDFs: {extraction_stats['total_pdfs']}")
    print(f"  Successful: {extraction_stats['successful_extractions']}")
    print(f"  Failed: {extraction_stats['failed_extractions']}")
    print(f"  Avg text length: {extraction_stats['avg_text_length']:.0f} chars")
    print(f"{'='*80}\n")
    
    # Update extraction_results
    current_results = state.get("extraction_results", {})
    current_results["extraction_stats"] = extraction_stats
    
    return {
        "papers_with_text": papers_with_text,
        "extraction_results": current_results
    }


# ========================
# NODE 3: FINALIZE
# ========================

def finalize_extraction_node(state: ExtractionState) -> Dict:
    """
    Finalize and save extraction results
    """
    print("\n" + "="*80)
    print("EXTRACTION STAGE 3: FINALIZATION")
    print("="*80)
    
    papers_with_text = state.get("papers_with_text", [])
    extraction_results = state.get("extraction_results", {})
    
    fetch_stats = extraction_results.get("fetch_stats", {})
    extraction_stats = extraction_results.get("extraction_stats", {})
    
    # Prepare output data
    output_data = {
        "metadata": {
            "extraction_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_papers": fetch_stats.get("total_papers", 0),
            "pdfs_fetched": fetch_stats.get("arxiv_direct", 0) + fetch_stats.get("arxiv_search", 0) + fetch_stats.get("open_access", 0),
            "text_extracted": extraction_stats.get("successful_extractions", 0),
            "extraction_method": "pymupdf_full_text",
            "fetch_sources": {
                "arxiv_direct": fetch_stats.get("arxiv_direct", 0),
                "arxiv_search": fetch_stats.get("arxiv_search", 0),
                "open_access": fetch_stats.get("open_access", 0)
            },
            "failed_fetches": fetch_stats.get("failed", 0),
            "failure_reasons": fetch_stats.get("failure_reasons", [])[:10]  # Limit to first 10
        },
        "papers": papers_with_text
    }
    
    print(f"\nFinal Extraction Statistics:")
    print(f"  Input papers: {output_data['metadata']['total_papers']}")
    print(f"  PDFs fetched: {output_data['metadata']['pdfs_fetched']}")
    print(f"  Text extracted: {output_data['metadata']['text_extracted']}")
    print(f"  Success rate: {(output_data['metadata']['text_extracted'] / max(output_data['metadata']['total_papers'], 1) * 100):.1f}%")
    
    # Save to file
    try:
        with open("extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved extracted_data.json ({len(papers_with_text)} papers)")
    except Exception as e:
        print(f"\n[FAIL] Error saving file: {e}")
    
    print(f"{'='*80}\n")
    
    return {
        "papers_with_text": papers_with_text,
        "extraction_results": extraction_results
    }


# ========================
# GRAPH ASSEMBLY
# ========================

workflow = StateGraph(ExtractionState)

workflow.add_node("fetch_pdfs", fetch_pdfs_node)
workflow.add_node("extract_text", extract_text_node)
workflow.add_node("finalize_extraction", finalize_extraction_node)

workflow.set_entry_point("fetch_pdfs")
workflow.add_edge("fetch_pdfs", "extract_text")
workflow.add_edge("extract_text", "finalize_extraction")
workflow.add_edge("finalize_extraction", END)

extraction_agent = workflow.compile()