import os
import json
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0
)

MAX_PAPERS_PER_BATCH = 3
MAX_CHARS_PER_PAPER = 1500
MIN_SUPPORT_PER_THEME = 3


# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class ThematicAnalysisState(TypedDict):
    extracted_data: Dict
    research_questions: List[str]

    aggregated_text: List[Dict]
    initial_themes: List[Dict]

    themes: List[Dict]
    thematic_results: Dict


# -------------------------------------------------------------------
# OUTPUT MODELS
# -------------------------------------------------------------------

class Theme(BaseModel):
    name: str
    description: str
    findings: List[str]


class ThemeList(BaseModel):
    themes: List[Theme]


class Attribution(BaseModel):
    theme_name: str
    citations: List[str]


class AttributionList(BaseModel):
    attributions: List[Attribution]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def truncate(text: str) -> str:
    return text[:MAX_CHARS_PER_PAPER] + "..." if len(text) > MAX_CHARS_PER_PAPER else text


def batch_items(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# -------------------------------------------------------------------
# NODES
# -------------------------------------------------------------------

def prepare_data_node(state: ThematicAnalysisState) -> Dict:
    print("\n" + "=" * 60)
    print("THEMATIC STAGE 1: DATA PREPARATION")
    print("=" * 60)

    extracted = state.get("extracted_data", {})
    papers = extracted.get("papers") or extracted.get("papers_with_text", [])

    aggregated = []

    for i, paper in enumerate(papers):
        paper_id = paper.get("paperId", f"paper_{i}")
        title = paper.get("title", "Untitled")

        text = ""
        sections = paper.get("extracted_text", {})

        if isinstance(sections, dict):
            for k in ["results", "discussion", "conclusion"]:
                if sections.get(k):
                    text += f"{k.upper()}:\n{sections[k]}\n\n"

        if not text:
            text = paper.get("abstract", "")

        aggregated.append({
            "paper_id": paper_id,
            "title": title,
            "content": truncate(text)
        })

    print(f"Prepared {len(aggregated)} papers.")
    return {"aggregated_text": aggregated}


# -------------------------------------------------------------------
# PHASE 1: THEME DISCOVERY
# -------------------------------------------------------------------

def identify_themes_node(state: ThematicAnalysisState) -> Dict:
    print("\n" + "=" * 60)
    print("THEMATIC STAGE 2: IDENTIFYING THEMES")
    print("=" * 60)

    papers = state["aggregated_text"][:MAX_PAPERS_PER_BATCH]
    rq = state["research_questions"]

    text_blob = "\n---\n".join(
        f"Title: {p['title']}\nContent: {p['content']}" for p in papers
    )

    parser = PydanticOutputParser(pydantic_object=ThemeList)

    prompt = f"""
You are conducting a systematic literature review.

Research Questions:
{chr(10).join(f"- {q}" for q in rq)}

Identify 6–10 recurring, well-defined themes.
Themes must be specific but applicable across multiple papers.

{text_blob}

CRITICAL:
- Output MUST be valid JSON only.
- Do NOT include explanations or markdown.

{parser.get_format_instructions()}
"""

    response = llm.invoke(prompt)
    parsed = parser.parse(response.content)

    themes = [t.model_dump() for t in parsed.themes]

    print(f"Identified {len(themes)} candidate themes.")
    return {"initial_themes": themes}


# -------------------------------------------------------------------
# PHASE 2: EVIDENCE ATTRIBUTION (FIXED)
# -------------------------------------------------------------------

def refine_themes_node(state: ThematicAnalysisState) -> Dict:
    print("\n" + "=" * 60)
    print("THEMATIC STAGE 3: ATTRIBUTING EVIDENCE")
    print("=" * 60)

    themes = state["initial_themes"]
    papers = state["aggregated_text"]

    citation_map = {t["name"]: set() for t in themes}
    parser = PydanticOutputParser(pydantic_object=AttributionList)

    for batch in batch_items(papers, MAX_PAPERS_PER_BATCH):
        batch_text = "\n---\n".join(
            f"Paper ID: {p['paper_id']}\nTitle: {p['title']}\nContent: {p['content']}"
            for p in batch
        )

        prompt = f"""
You are performing STRICT evidence attribution.

CRITICAL RULES (MANDATORY):
- Output MUST be valid JSON only.
- Do NOT include explanations, summaries, markdown, or prose.
- Do NOT repeat paper content.
- Do NOT include text outside the JSON object.
- If no papers support a theme, return an empty list.

JSON schema:
{parser.get_format_instructions()}

Themes:
{json.dumps(themes, indent=2)}

Papers:
{batch_text}
"""

        response = llm.invoke(prompt)

        try:
            parsed = parser.parse(response.content)
            attributions = parsed.attributions
        except Exception:
            print("⚠️ Attribution parsing failed for this batch. Skipping batch.")
            print(response.content[:500])
            continue

        for att in attributions:
            if att.theme_name in citation_map:
                citation_map[att.theme_name].update(att.citations)

    refined = []
    for t in themes:
        citations = sorted(citation_map[t["name"]])
        if len(citations) >= MIN_SUPPORT_PER_THEME:
            t["citations"] = citations
            refined.append(t)

    print(f"Retained {len(refined)} themes after support filtering.")
    return {"themes": refined}


# -------------------------------------------------------------------
# FINALIZATION
# -------------------------------------------------------------------

def finalize_analysis_node(state: ThematicAnalysisState) -> Dict:
    print("\n" + "=" * 60)
    print("THEMATIC STAGE 4: FINALIZATION")
    print("=" * 60)

    return {
        "themes": state.get("themes", []),
        "thematic_results": {
            "total_themes": len(state.get("themes", [])),
            "input_papers": len(state.get("aggregated_text", [])),
            "min_support": MIN_SUPPORT_PER_THEME,
            "method": "Batch-wise LLM thematic analysis with defensive parsing"
        }
    }


# -------------------------------------------------------------------
# GRAPH ASSEMBLY
# -------------------------------------------------------------------

workflow = StateGraph(ThematicAnalysisState)

workflow.add_node("prepare_data", prepare_data_node)
workflow.add_node("identify_themes", identify_themes_node)
workflow.add_node("refine_themes", refine_themes_node)
workflow.add_node("finalize", finalize_analysis_node)

workflow.set_entry_point("prepare_data")
workflow.add_edge("prepare_data", "identify_themes")
workflow.add_edge("identify_themes", "refine_themes")
workflow.add_edge("refine_themes", "finalize")
workflow.add_edge("finalize", END)

thematic_analysis_agent = workflow.compile()
