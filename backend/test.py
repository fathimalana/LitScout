import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Agents
from backend.agents.search_and_filter_agent import saf_agent, REPUTED_SOURCES
from backend.agents.screening_agent import screening_agent
from backend.agents.extraction_agent import extraction_agent
from backend.agents.thematic_analysis_agent import thematic_analysis_agent

# 🔹 NEW: adaptive extraction
from backend.agents.adaptive_extraction import adaptive_extraction_loop


def test_full_pipeline():
    """
    Tests the entire LitScout pipeline:
    1. Search & Filter
    2. Screening
    3. Adaptive Extraction
    4. Thematic Analysis
    """
    print("\n" + "=" * 80)
    print("TESTING FULL LITSCOUT PIPELINE")
    print("=" * 80)

    # --- 0. INITIAL CONFIGURATION ---
    research_questions = [
        "What are the recent advancements in multi-agent reinforcement learning?",
        "How do autonomous agents coordinate in dynamic environments?"
    ]
    inclusion_criteria = "Focus on deep reinforcement learning and cooperative settings."
    exclusion_criteria = "Exclude purely theoretical game theory papers without RL applications."

    current_year = datetime.now().year
    start_year = current_year - 2
    end_year = current_year

    print("\n[CONFIG] Research Questions:")
    for q in research_questions:
        print(f"  - {q}")
    print(f"[CONFIG] Timeframe: {start_year}-{end_year}")

    # --- 1. SEARCH AND FILTER AGENT ---
    print("\n" + "-" * 40)
    print("STEP 1: SEARCH & FILTER AGENT")
    print("-" * 40)

    saf_input = {
        "research_questions": research_questions,
        "start_year": start_year,
        "end_year": end_year,
        "sources": REPUTED_SOURCES,
        "raw_papers": [],
        "filtered_papers": []
    }

    print("Running Search & Filter Agent...")
    saf_result = saf_agent.invoke(saf_input)

    filtered_papers = saf_result.get("filtered_papers", [])
    print(f"✅ STEP 1 COMPLETE: Found {len(filtered_papers)} papers after filtering.")

    if not filtered_papers:
        print("❌ Pipeline stopped: No papers found.")
        return

    # --- 2. SCREENING AGENT ---
    print("\n" + "-" * 40)
    print("STEP 2: SCREENING AGENT")
    print("-" * 40)

    screening_input = {
        "filtered_papers": filtered_papers,
        "research_questions": research_questions,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "keyword_high_threshold": 0.0,
        "keyword_medium_threshold": 0.0,
        "tfidf_threshold": 0.0,
        "use_llm_screening": False,
        "max_papers": 0,
        "high_relevance_papers": [],
        "medium_relevance_papers": [],
        "borderline_papers": [],
        "screened_papers": [],
        "screening_results": {}
    }

    print(f"Running Screening Agent on {len(filtered_papers)} papers...")
    screening_result = screening_agent.invoke(screening_input)

    screened_papers = screening_result.get("screened_papers", [])
    print(f"✅ STEP 2 COMPLETE: {len(screened_papers)} papers passed screening.")

    if not screened_papers:
        print("❌ Pipeline stopped: All papers screened out.")
        return

    # --- 3. ADAPTIVE EXTRACTION ---
    print("\n" + "-" * 40)
    print("STEP 3: ADAPTIVE EXTRACTION")
    print("-" * 40)

    print(f"Running adaptive extraction on {len(screened_papers)} screened papers...")

    extracted_papers, extraction_metadata = adaptive_extraction_loop(
        screened_papers=screened_papers,
        research_questions=research_questions,
        extraction_agent=extraction_agent
    )

    print(f"✅ STEP 3 COMPLETE: Extracted text from {len(extracted_papers)} papers.")

    if not extracted_papers:
        print("❌ Pipeline stopped: Could not extract text from any papers.")
        return

    # Prepare data in the SAME shape expected by thematic agent
    extraction_data = {
        "papers": extracted_papers
    }

    # --- 4. THEMATIC ANALYSIS AGENT ---
    print("\n" + "-" * 40)
    print("STEP 4: THEMATIC ANALYSIS AGENT")
    print("-" * 40)

    thematic_input = {
        "extracted_data": extraction_data,
        "research_questions": research_questions,
        "aggregated_text": [],
        "initial_themes": [],
        "themes": [],
        "thematic_results": {}
    }

    print("Running Thematic Analysis Agent...")
    thematic_result = thematic_analysis_agent.invoke(thematic_input)

    themes = thematic_result.get("themes", [])
    print(f"✅ STEP 4 COMPLETE: Identified {len(themes)} themes.")

    # --- 5. RESULTS ---
    print("\n" + "=" * 80)
    print("PIPELINE TEST RESULTS")
    print("=" * 80)

    for i, theme in enumerate(themes, 1):
        print(f"\nTHEME {i}: {theme.get('name', 'Untitled')}")
        print(f"Description: {theme.get('description', 'No description')}")
        print("Findings:")
        for finding in theme.get("findings", [])[:3]:
            print(f"  - {finding}")
        print(f"Supported by {len(theme.get('citations', []))} papers")

    print("\n✅ FULL PIPELINE TEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    test_full_pipeline()
