import os
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. CONFIGURATION ---
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY,    
    temperature=0
)

# --- 2. STATE DEFINITION ---
class ScreeningState(TypedDict):
    # Input
    filtered_papers: List[Dict]
    research_questions: List[str]
    inclusion_criteria: str
    exclusion_criteria: str
    
    # ADAPTIVE: LLM determines these
    keyword_high_threshold: float
    keyword_medium_threshold: float
    tfidf_threshold: float
    use_llm_screening: bool
    
    # Internal stages
    high_relevance_papers: List[Dict]
    medium_relevance_papers: List[Dict]
    borderline_papers: List[Dict]
    
    # Final output
    screened_papers: List[Dict]
    screening_results: Dict


# --- 3. THRESHOLD ANALYSIS ---

class ThresholdRecommendation(BaseModel):
    """LLM output for threshold recommendations."""
    keyword_high_threshold: float = Field(description="Threshold for high relevance (0.0-1.0)")
    keyword_medium_threshold: float = Field(description="Threshold for medium relevance (0.0-1.0)")
    tfidf_threshold: float = Field(description="TF-IDF similarity threshold (0.0-1.0)")
    use_llm_screening: bool = Field(description="Whether to use LLM for borderline papers")
    reasoning: str = Field(description="Explanation for chosen thresholds")


# --- 4. HELPER FUNCTIONS ---

def save_screened_papers(papers: List[Dict], filename: str = "screened_papers.txt"):
    """Save screened papers to a text file for review."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SCREENED PAPERS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Total Papers: {len(papers)}\n")
            f.write("="*80 + "\n\n")
            
            for i, paper in enumerate(papers, 1):
                if paper is None:
                    continue
                    
                f.write(f"Paper {i}\n")
                f.write("-"*80 + "\n")
                f.write(f"ID: {paper.get('paperId', 'N/A')}\n")
                f.write(f"Title: {paper.get('title', 'No title')}\n")
                f.write(f"Year: {paper.get('year', 'N/A')}\n")
                
                authors = paper.get('authors', [])
                if authors:
                    author_names = [a.get('name', 'Unknown') for a in authors[:3]]
                    f.write(f"Authors: {', '.join(author_names)}")
                    if len(authors) > 3:
                        f.write(f" et al. ({len(authors)} total)")
                    f.write("\n")
                
                abstract = paper.get('abstract', '')
                if abstract:
                    f.write(f"Abstract: {abstract}\n")
                else:
                    f.write("Abstract: Not available\n")
                
                f.write(f"Venue: {paper.get('venue', 'N/A')}\n")
                f.write(f"Citations: {paper.get('citationCount', 0)}\n")
                
                pub_types = paper.get('publicationTypes', [])
                if pub_types:
                    f.write(f"Type: {', '.join(pub_types)}\n")
                
                f.write("\n")
        
        print(f"Saved {len(papers)} screened papers to {filename}")
        
    except Exception as e:
        print(f"Warning: Could not save papers to file: {e}")


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return {w for w in text.split() if len(w) > 3}


def calculate_keyword_score(paper: Dict, research_questions: List[str], inclusion_criteria: str) -> float:
    """Calculate keyword overlap score."""
    paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    if not paper_text.strip():
        return 0.0
    
    context_text = " ".join(research_questions) + " " + inclusion_criteria
    
    paper_kw = extract_keywords(paper_text)
    context_kw = extract_keywords(context_text)
    
    if not context_kw:
        return 0.5
    
    overlap = len(paper_kw.intersection(context_kw))
    return min(overlap / len(context_kw), 1.0)


def analyze_and_set_thresholds_node(state: ScreeningState) -> dict:
    """
    Stage 1: Analyze actual score distribution and set adaptive thresholds.
    """
    print("\n" + "="*60)
    print("STAGE 1: ADAPTIVE THRESHOLD ANALYSIS")
    print("="*60)
    
    research_questions = state["research_questions"]
    inclusion_criteria = state["inclusion_criteria"]
    papers = state["filtered_papers"]
    
    # Sample papers for analysis
    sample_size = min(200, len(papers))
    sample_papers = [p for p in papers[:sample_size] if p is not None]
    
    print(f"Analyzing {len(sample_papers)} sample papers to determine score distribution...")
    
    # Calculate actual score distribution FIRST
    sample_scores = []
    for paper in sample_papers:
        score = calculate_keyword_score(paper, research_questions, inclusion_criteria)
        sample_scores.append(score)
    
    if not sample_scores:
        print("Warning: No valid scores, using conservative defaults")
        return {
            "keyword_high_threshold": 0.15,
            "keyword_medium_threshold": 0.08,
            "tfidf_threshold": 0.3,
            "use_llm_screening": True
        }
    
    # Calculate percentiles for data-driven thresholds
    max_score = max(sample_scores)
    mean_score = np.mean(sample_scores)
    p75 = np.percentile(sample_scores, 75)
    p50 = np.percentile(sample_scores, 50)
    p25 = np.percentile(sample_scores, 25)
    
    print(f"\nSample Score Distribution (n={len(sample_scores)}):")
    print(f"   Max: {max_score:.3f}")
    print(f"   75th percentile: {p75:.3f}")
    print(f"   Median: {p50:.3f}")
    print(f"   25th percentile: {p25:.3f}")
    print(f"   Mean: {mean_score:.3f}")
    
    # Extract sample titles for LLM context
    sample_titles = [p.get('title', '')[:100] for p in sample_papers[:10]]
    
    # Create analysis prompt with ACTUAL score statistics
    prompt = f"""You are an expert research methodology consultant. Based on ACTUAL score distribution data, recommend optimal screening thresholds.

Research Questions:
{chr(10).join(f"- {q}" for q in research_questions)}

Inclusion Criteria: {inclusion_criteria}

ACTUAL SCORE STATISTICS from {len(sample_scores)} papers:
- Maximum score: {max_score:.3f}
- 75th percentile: {p75:.3f}
- Median: {p50:.3f}
- Mean: {mean_score:.3f}
- 25th percentile: {p25:.3f}

Sample Titles:
{chr(10).join(f"- {t}" for t in sample_titles[:5])}

Total Papers: {len(papers)}

IMPORTANT: Thresholds MUST be based on ACTUAL scores above, not theoretical values!

Strategy:
- If max_score < 0.35: Use LENIENT thresholds (scores are low)
- If max_score 0.35-0.5: Use MODERATE thresholds
- If max_score > 0.5: Use STRICTER thresholds

keyword_high_threshold: Auto-include papers above this
- Should be 60-70% of max_score OR around 75th percentile
- Example: if max=0.342, use 0.22-0.24

keyword_medium_threshold: Papers need TF-IDF review
- Should be 40-50% of max_score OR around median
- Example: if max=0.342, use 0.14-0.17

tfidf_threshold: Semantic similarity threshold
- Use 0.35-0.45 for low max_scores (< 0.4)
- Use 0.45-0.60 for higher max_scores (> 0.4)

use_llm_screening: Use LLM for borderline cases?
- true if: dataset < 500 papers
- false if: dataset > 2000 papers

Respond with JSON only:
{{
  "keyword_high_threshold": <float>,
  "keyword_medium_threshold": <float>,
  "tfidf_threshold": <float>,
  "use_llm_screening": <bool>,
  "reasoning": "<brief explanation>"
}}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        recommendation = json.loads(content)
        
        # Safety checks
        rec_high = recommendation['keyword_high_threshold']
        rec_medium = recommendation['keyword_medium_threshold']
        
        # Override if too high
        if rec_high > max_score * 0.8:
            print(f"Warning: LLM's high threshold ({rec_high:.3f}) too high, adjusting to {max_score * 0.68:.3f}")
            rec_high = max(max_score * 0.68, p75)
        
        if rec_medium > max_score * 0.55:
            print(f"Warning: LLM's medium threshold ({rec_medium:.3f}) too high, adjusting to {max_score * 0.40:.3f}")
            rec_medium = max(max_score * 0.40, p50)
        
        # Ensure high > medium
        if rec_high <= rec_medium:
            rec_high = rec_medium + 0.05
        
        print(f"\nFinal Adaptive Thresholds:")
        print(f"   Keyword High: {rec_high:.3f} (auto-include)")
        print(f"   Keyword Medium: {rec_medium:.3f} (needs TF-IDF)")
        print(f"   TF-IDF: {recommendation['tfidf_threshold']:.3f}")
        print(f"   Use LLM: {recommendation['use_llm_screening']}")
        print(f"\n{recommendation['reasoning']}")
        
        return {
            "keyword_high_threshold": rec_high,
            "keyword_medium_threshold": rec_medium,
            "tfidf_threshold": recommendation['tfidf_threshold'],
            "use_llm_screening": recommendation['use_llm_screening']
        }
        
    except Exception as e:
        print(f"Warning: Error in threshold analysis: {e}")
        
        # Fallback to percentile-based thresholds
        high_threshold = max(p75, max_score * 0.68)
        medium_threshold = max(p50, max_score * 0.40)
        
        print(f"Using percentile-based fallback:")
        print(f"   High: {high_threshold:.3f}")
        print(f"   Medium: {medium_threshold:.3f}")
        
        return {
            "keyword_high_threshold": high_threshold,
            "keyword_medium_threshold": medium_threshold,
            "tfidf_threshold": 0.4,
            "use_llm_screening": len(papers) < 1000
        }


# --- 5. SCREENING NODES ---

def keyword_screening_node(state: ScreeningState) -> dict:
    """Stage 2: Keyword screening with adaptive thresholds."""
    print("\n" + "="*60)
    print("STAGE 2: ADAPTIVE KEYWORD SCREENING")
    print("="*60)
    
    papers = state["filtered_papers"]
    research_questions = state.get("research_questions", [])
    inclusion_criteria = state.get("inclusion_criteria", "")
    
    # Use adaptive thresholds
    high_threshold = state.get("keyword_high_threshold", 0.3)
    medium_threshold = state.get("keyword_medium_threshold", 0.15)
    
    valid_papers = [p for p in papers if p is not None]
    print(f"Processing {len(valid_papers)} papers...")
    print(f"Adaptive Thresholds: High={high_threshold:.2f}, Medium={medium_threshold:.2f}")
    
    high = []
    medium = []
    excluded = 0
    scores = []
    
    for paper in valid_papers:
        score = calculate_keyword_score(paper, research_questions, inclusion_criteria)
        scores.append(score)
        
        if score >= high_threshold:
            high.append(paper)
        elif score >= medium_threshold:
            medium.append(paper)
        else:
            excluded += 1
            with open("excluded_papers_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Score: {score:.3f} | ID: {paper.get('paperId')} | Title: {paper.get('title')}\n")
    
    if scores:
        print(f"Score stats: Max={max(scores):.3f}, Mean={np.mean(scores):.3f}, Min={min(scores):.3f}")
    
    print(f"High relevance: {len(high)}")
    print(f"Medium relevance: {len(medium)}")
    print(f"Excluded: {excluded}")
    
    return {
        "high_relevance_papers": high,
        "medium_relevance_papers": medium
    }


def tfidf_screening_node(state: ScreeningState) -> dict:
    """Stage 3: TF-IDF screening with adaptive threshold."""
    print("\n" + "="*60)
    print("STAGE 3: ADAPTIVE TF-IDF SCREENING")
    print("="*60)
    
    medium_papers = state["medium_relevance_papers"]
    research_questions = state["research_questions"]
    threshold = state.get("tfidf_threshold", 0.5)
    
    if not medium_papers:
        print("No papers to process.")
        return {"borderline_papers": []}
    
    print(f"Processing {len(medium_papers)} papers...")
    print(f"Adaptive TF-IDF Threshold: {threshold:.2f}")
    
    paper_texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in medium_papers]
    query_text = " ".join(research_questions)
    
    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        all_texts = paper_texts + [query_text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        query_vector = tfidf_matrix[-1]
        paper_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(paper_vectors, query_vector).flatten()
        
        print(f"Similarity stats: Max={max(similarities):.3f}, Mean={np.mean(similarities):.3f}")
        
        promoted = []
        borderline = []
        
        for paper, similarity in zip(medium_papers, similarities):
            if similarity >= threshold:
                promoted.append(paper)
            elif similarity >= (threshold - 0.15):
                borderline.append(paper)
        
        print(f"Promoted: {len(promoted)}")
        print(f"Borderline: {len(borderline)}")
        
        updated_high = state["high_relevance_papers"] + promoted
        
        return {
            "high_relevance_papers": updated_high,
            "borderline_papers": borderline
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"borderline_papers": medium_papers}


def llm_screening_node(state: ScreeningState) -> dict:
    """Stage 4: LLM screening (conditional)."""
    print("\n" + "="*60)
    print("STAGE 4: CONDITIONAL LLM SCREENING")
    print("="*60)
    
    use_llm = state.get("use_llm_screening", True)
    borderline = state["borderline_papers"]
    
    if not use_llm:
        print("LLM screening disabled by adaptive analysis.")
        return {}
    
    if not borderline:
        print("No borderline papers.")
        return {}
    
    if len(borderline) > 100:
        print(f"Too many borderline papers ({len(borderline)}). Taking top 100.")
        borderline = borderline[:100]
    
    print(f"Using LLM for {len(borderline)} papers...")
    
    research_questions = state["research_questions"]
    inclusion_criteria = state["inclusion_criteria"]
    approved = []
    batch_size = 8
    
    for i in range(0, len(borderline), batch_size):
        batch = borderline[i:i+batch_size]
        
        papers_text = "\n\n".join([
            f"[{idx+1}] {p.get('title', 'N/A')}\n{p.get('abstract', 'N/A')[:500]}"
            for idx, p in enumerate(batch)
        ])
        
        prompt = f"""Screen papers for literature review.

Questions: {'; '.join(research_questions)}
Criteria: {inclusion_criteria}

Output ONLY "1" (relevant) or "0" (not relevant) per paper, one per line.

Papers:
{papers_text}

Output:"""

        try:
            response = llm.invoke(prompt)
            lines = response.content.strip().split('\n')
            
            for paper, line in zip(batch, lines):
                if '1' in line:
                    approved.append(paper)
                    
        except Exception as e:
            print(f"Batch error: {e}")
            approved.extend(batch)
    
    print(f"Approved: {len(approved)}")
    
    final_high = state["high_relevance_papers"] + approved
    
    return {"high_relevance_papers": final_high}


def finalize_node(state: ScreeningState) -> dict:
    """Stage 5: Finalize and save results."""
    print("\n" + "="*60)
    print("STAGE 5: FINALIZING")
    print("="*60)
    
    screened = state["high_relevance_papers"]
    
    # Deduplicate
    seen = set()
    unique = []
    for paper in screened:
        if paper is None:
            continue
        pid = paper.get('paperId', id(paper))
        if pid not in seen:
            seen.add(pid)
            unique.append(paper)
    
    # Sort by year and citations
    screened_papers = sorted(
        unique, 
        key=lambda p: (p.get('year', 0), p.get('citationCount', 0)), 
        reverse=True
    )
    
    results = {
        "total_input_papers": len(state["filtered_papers"]),
        "total_screened_papers": len(screened_papers),
        "adaptive_thresholds_used": {
            "keyword_high": state.get("keyword_high_threshold", 0),
            "keyword_medium": state.get("keyword_medium_threshold", 0),
            "tfidf": state.get("tfidf_threshold", 0)
        },
        "llm_used": state.get("use_llm_screening", False),
        "exclusion_rate": 1 - (len(screened_papers) / max(len(state["filtered_papers"]), 1)),
        "screening_method": "adaptive_pipeline"
    }
    
    print(f"\n{'='*60}")
    print(f"Input: {results['total_input_papers']} -> Output: {results['total_screened_papers']}")
    print(f"Exclusion: {results['exclusion_rate']:.1%}")
    print(f"{'='*60}\n")
    
    # Save to file
    save_screened_papers(screened_papers)
    
    return {
        "screened_papers": screened_papers,
        "screening_results": results
    }


# --- 6. GRAPH ASSEMBLY ---

workflow = StateGraph(ScreeningState)

workflow.add_node("analyze_thresholds", analyze_and_set_thresholds_node)
workflow.add_node("keyword_screening", keyword_screening_node)
workflow.add_node("tfidf_screening", tfidf_screening_node)
workflow.add_node("llm_screening", llm_screening_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("analyze_thresholds")
workflow.add_edge("analyze_thresholds", "keyword_screening")
workflow.add_edge("keyword_screening", "tfidf_screening")
workflow.add_edge("tfidf_screening", "llm_screening")
workflow.add_edge("llm_screening", "finalize")
workflow.add_edge("finalize", END)

screening_agent = workflow.compile()