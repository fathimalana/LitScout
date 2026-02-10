import os
import ast
import re
import time
from datetime import datetime
import requests
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION ---
load_dotenv()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

if not SEMANTIC_SCHOLAR_API_KEY :
    raise ValueError("One or more required API keys are not set in the .env file.")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,    
    temperature=0
)

# --- CONSTANTS ---
REPUTED_SOURCES = [
    "Springer",
    "Elsevier",
    "ScienceDirect",
    "ACM",
    "IEEE",
    "Nature",
    "Wiley",
    "Taylor & Francis",
    "Sage",
    "Oxford University Press",
    "Cambridge University Press",
    "AAAI",
    "PLOS",
    "Science",
    "PNAS" # Proceedings of the National Academy of Sciences
]

# --- 2. QUERY GENERATION LOGIC ---
def generate_simple_query(research_question: str) -> str:
    """Uses an LLM to distill a research question into a simple keyword query."""
    prompt = (
        "You are an expert academic researcher. Rephrase the following research question "
        "into a simple, effective keyword query for a search engine like Google Scholar. "
        "Use only the 3-5 most important keywords or quoted phrases. "
        "Do not use Boolean operators like AND/OR. "
        f"Research Question: '{research_question}'"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip().replace('"', '')
    except Exception as e:
        print(f"Error during query generation: {e}")
        return research_question


# --- 3. SEARCH AND FILTERING LOGIC ---
def paper_search(state: dict) -> dict:
    """The main search and filter node for the sub-graph."""
    print("---SUB-AGENT: Starting paper search and filter process---")
    
    research_questions = state.get("research_questions", [])
    start_year = state.get("start_year", datetime.now().year - 5)
    end_year = state.get("end_year", datetime.now().year)
    
    # Use provided sources, or default to REPUTED_SOURCES if explicitly requested/configured
    # For now, we respect the 'sources' passed in state. 
    # If the orchestrator wants to use REPUTED_SOURCES, it should pass them.
    sources_to_include = state.get("sources", [])
    
    all_unique_papers = {}

    for question in research_questions:
        print(f"\nProcessing question: {question}")
        simple_query = generate_simple_query(question)
        print(f"Simplified query: {simple_query}")
        
        current_offset = 0
        limit_per_request = 100
        
        while True:
            params = {
                'query': simple_query,
                'fields': 'title,abstract,authors,year,url,venue',
                'limit': limit_per_request,
                'offset': current_offset
            }
            try:
                response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers={'x-api-key': SEMANTIC_SCHOLAR_API_KEY})
                response.raise_for_status()
                results = response.json()
                batch_papers = results.get('data', [])
                
                if not batch_papers: break
                
                for paper in batch_papers:
                    all_unique_papers[paper['paperId']] = paper
                
                print(f"Fetched {len(batch_papers)} papers. Total unique papers now: {len(all_unique_papers)}")
                
                current_offset += limit_per_request
                if current_offset >= results.get('total', 0) or current_offset >= 1000:
                    break
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"Error during paper search: {e}")
                break
        time.sleep(2)

    # --- APPLY FILTERING LOGIC ---
    print(f"\n---SUB-AGENT: Applying filters to {len(all_unique_papers)} papers---")
    
    papers_to_filter = list(all_unique_papers.values())
    
    # Filter 1: Abstract
    papers_with_abstracts = [p for p in papers_to_filter if p.get('abstract')]
    print(f"  - {len(papers_with_abstracts)} papers remaining after abstract filter.")
    
    # Filter 2: Date Range
    papers_in_date_range = [
        p for p in papers_with_abstracts 
        if p.get('year') and start_year <= p['year'] <= end_year
    ]
    print(f"  - {len(papers_in_date_range)} papers remaining after date filter ({start_year}-{end_year}).")

    # --- NEW: More Lenient Source Filter ---
    # If sources are provided, we filter by them.
    if sources_to_include:
        filtered_by_source = []
        for paper in papers_in_date_range:
            venue = paper.get('venue', '') or ''
            
            # If the venue is missing, we keep the paper by default (lenient)
            if not venue:
                filtered_by_source.append(paper)
                continue
            
            # If the venue exists, we check if it matches our list
            if any(source.lower() in venue.lower() for source in sources_to_include):
                filtered_by_source.append(paper)
        
        print(f"  - {len(filtered_by_source)} papers remaining after source filter.")
        final_paper_list = filtered_by_source
    else:
        # No source filter applied
        final_paper_list = papers_in_date_range

    print(f"---SUB-AGENT: Found a total of {len(final_paper_list)} filtered, unique papers---")
    
    return {"raw_papers": list(all_unique_papers.values()), "filtered_papers": final_paper_list}


# --- 4. SUB-GRAPH DEFINITION ---
class SearchFilterState(TypedDict):
    research_questions: List[str]
    start_year: Optional[int]
    end_year: Optional[int]
    sources: Optional[List[str]]
    raw_papers: List[Dict]
    filtered_papers: List[Dict]

workflow = StateGraph(SearchFilterState)
workflow.add_node("paper_search", paper_search)
workflow.set_entry_point("paper_search")
workflow.add_edge("paper_search", END)

saf_agent = workflow.compile()