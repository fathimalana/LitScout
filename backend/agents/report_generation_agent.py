import os
import json
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.2,   # slight creativity for report prose
)

SECTION_BATCH_SIZE = 2   # sections drafted per LLM call to stay within rate limits


# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class ReportState(TypedDict):
    # Inputs (from orchestrator)
    synthesis_draft:    str
    themes:             List[Dict]
    research_questions: List[str]
    user_prompt:        str
    quality_report:     Dict

    # Intermediate
    report_plan:        List[Dict]    # list of {title, instructions} section specs
    section_drafts:     List[Dict]    # list of {title, content} drafted sections

    # Outputs
    final_report:       str
    report_metadata:    Dict


# -------------------------------------------------------------------
# PYDANTIC OUTPUT MODELS
# -------------------------------------------------------------------

class ReportSection(BaseModel):
    title:   str = Field(description="Section title.")
    content: str = Field(description="Full section prose — plain academic text, no markdown symbols.")

class ReportSectionList(BaseModel):
    sections: List[ReportSection]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _batch(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]

def _quality_summary(qr: Dict) -> str:
    if not qr:
        return "Quality assessment not available."
    return (
        f"Overall score: {qr.get('score', 'N/A')}/10. "
        f"RQ Coverage: {qr.get('coverage_score', 0.0):.0%}. "
        f"Inline citations: {qr.get('citation_count', 0)}. "
        f"Status: {qr.get('status', 'unknown')}. "
        f"{qr.get('reasoning', '')}"
    )


# -------------------------------------------------------------------
# NODE 1 — PLAN REPORT STRUCTURE
# -------------------------------------------------------------------

def plan_report_node(state: ReportState) -> Dict:
    print("\n" + "=" * 60)
    print("REPORT STAGE 1: PLANNING STRUCTURE")
    print("=" * 60)

    themes = state.get("themes", [])
    rqs    = state.get("research_questions", [])
    qr     = state.get("quality_report", {})

    theme_names = [t.get("name", f"Theme {i+1}") for i, t in enumerate(themes)]

    # Build a fixed academic section plan
    plan = [
        {
            "title": "Executive Summary",
            "instructions": (
                f"Write a concise 3–4 sentence executive summary of this literature review. "
                f"State the research topic, number of themes identified ({len(themes)}), "
                f"and the main conclusion. Do NOT use bullet points."
            )
        },
        {
            "title": "Research Methodology",
            "instructions": (
                f"Describe the systematic review methodology used. Mention: multi-agent AI pipeline, "
                f"semantic paper search, abstract screening, PDF extraction, thematic analysis, and synthesis. "
                f"Reference the research questions: {'; '.join(rqs[:3])}. "
                f"Write 2–3 formal academic paragraphs."
            )
        },

        {
            "title": "Quality Assessment",
            "instructions": (
                f"Summarise the quality assessment of this review in 2 paragraphs. "
                f"Quality metrics: {_quality_summary(qr)}. "
                f"Discuss limitations, potential biases, and what could strengthen the review. "
                f"Suggestions: {qr.get('suggestions', [])}."
            )
        },
        {
            "title": "Conclusions and Future Directions",
            "instructions": (
                f"Write 3–4 sentences synthesising the overall conclusions of the review. "
                f"Then write 2–3 sentences on promising future research directions. "
                f"Address these research questions: {json.dumps(rqs)}."
            )
        },
        {
            "title": "References",
            "instructions": (
                "Extract ALL inline citations of the form [Author, Year] or [paperId] from the COMPLETE synthesis draft below. "
                "Format them as a numbered reference list — one entry per unique citation. "
                "List every citation found — do not skip any. "
                f"Synthesis text to scan:\n{state.get('synthesis_draft', '')[:6000]}\n"
                "If no citations found, write 'No formal references were automatically extracted.'"
            )
        }
    ]

    print(f"Planned {len(plan)} report sections.")
    return {"report_plan": plan}


# -------------------------------------------------------------------
# NODE 2 — DRAFT SECTIONS
# -------------------------------------------------------------------

def draft_sections_node(state: ReportState) -> Dict:
    print("\n" + "=" * 60)
    print("REPORT STAGE 2: DRAFTING SECTIONS")
    print("=" * 60)

    plan           = state.get("report_plan", [])
    synthesis      = state.get("synthesis_draft", "")
    user_prompt    = state.get("user_prompt", "")
    rqs            = state.get("research_questions", [])

    if not plan:
        print("⚠️  No report plan — skipping drafting.")
        return {"section_drafts": []}

    rq_text        = "\n".join(f"- {q}" for q in rqs)
    parser         = PydanticOutputParser(pydantic_object=ReportSectionList)
    all_sections: List[Dict] = []

    for batch_idx, batch in enumerate(_batch(plan, SECTION_BATCH_SIZE)):
        print(f"  Drafting batch {batch_idx + 1} ({len(batch)} sections)…")

        sections_spec = "\n\n".join(
            f"SECTION {i+1}: {s['title']}\nInstructions: {s['instructions']}"
            for i, s in enumerate(batch)
        )

        prompt = f"""You are an expert academic writer producing a formal literature review report.

Research Topic: {user_prompt}

Research Questions:
{rq_text}

Synthesis Draft (for reference):
{synthesis[:8000]}

Write EXACTLY {len(batch)} section(s) as specified below. Follow the instructions precisely.

{sections_spec}

CRITICAL RULES:
- Output MUST be valid JSON only.
- Plain academic prose — no **, no #, no markdown formatting inside content.
- Do NOT include text outside the JSON object.

{parser.get_format_instructions()}
"""
        try:
            response = llm.invoke(prompt)
            parsed   = parser.parse(response.content)
            for section in parsed.sections:
                all_sections.append({"title": section.title, "content": section.content})
        except Exception as e:
            print(f"  ⚠️  Batch {batch_idx + 1} parsing failed: {e}")
            # Fallback: add raw sections with placeholder content
            for s in batch:
                all_sections.append({
                    "title":   s["title"],
                    "content": f"[Content generation failed for this section. Error: {e}]"
                })

    print(f"Drafted {len(all_sections)} sections.")
    return {"section_drafts": all_sections}


# -------------------------------------------------------------------
# NODE 3 — ASSEMBLE REPORT
# -------------------------------------------------------------------

def assemble_report_node(state: ReportState) -> Dict:
    print("\n" + "=" * 60)
    print("REPORT STAGE 3: ASSEMBLING FINAL REPORT")
    print("=" * 60)

    section_drafts = state.get("section_drafts", [])
    user_prompt    = state.get("user_prompt", "")
    rqs            = state.get("research_questions", [])
    synthesis_draft = state.get("synthesis_draft", "")

    if not section_drafts:
        fallback = (
            "Report generation could not be completed: no sections were drafted. "
            "This may be due to missing synthesis output or API errors."
        )
        print("⚠️  No sections — returning fallback report.")
        return {"final_report": fallback}

    rq_text        = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(rqs))

    # Build the header
    header = (
        f"LITERATURE REVIEW REPORT\n"
        f"{'=' * 60}\n"
        f"Research Topic: {user_prompt}\n\n"
        f"Research Questions:\n{rq_text}\n"
        f"{'=' * 60}\n"
    )

    # Assemble sections with separators
    body_parts = []
    for section in section_drafts:
        title   = section.get("title", "Untitled Section")
        content = section.get("content", "").strip()
        body_parts.append(f"{title.upper()}\n{'-' * len(title)}\n{content}")
        
        # Inject the verbatim synthesis draft directly after methodology
        if "methodology" in title.lower():
            body_parts.append(f"KEY THEMES AND FINDINGS\n{'-' * 23}\n{synthesis_draft.strip()}")

    final_report = header + "\n\n".join(body_parts)

    print(f"Report assembled: {len(final_report.split())} words, {len(section_drafts)} sections.")
    return {"final_report": final_report}


# -------------------------------------------------------------------
# NODE 4 — FINALIZE
# -------------------------------------------------------------------

def finalize_report_node(state: ReportState) -> Dict:
    print("\n" + "=" * 60)
    print("REPORT STAGE 4: FINALIZING")
    print("=" * 60)

    report         = state.get("final_report", "")
    section_drafts = state.get("section_drafts", [])
    themes         = state.get("themes", [])
    quality_report = state.get("quality_report", {})

    word_count = len(report.split())

    metadata = {
        "word_count":      word_count,
        "section_count":   len(section_drafts),
        "theme_count":     len(themes),
        "quality_score":   quality_report.get("score", "N/A"),
        "method":          "LangGraph batch-draft report generation",
        "status":          "complete" if word_count > 200 else "partial"
    }

    print(f"Report finalised: {word_count} words across {len(section_drafts)} sections.")
    return {
        "final_report":   report,
        "report_metadata": metadata
    }


# -------------------------------------------------------------------
# GRAPH ASSEMBLY
# -------------------------------------------------------------------

workflow = StateGraph(ReportState)

workflow.add_node("plan_report",     plan_report_node)
workflow.add_node("draft_sections",  draft_sections_node)
workflow.add_node("assemble_report", assemble_report_node)
workflow.add_node("finalize_report", finalize_report_node)

workflow.set_entry_point("plan_report")
workflow.add_edge("plan_report",     "draft_sections")
workflow.add_edge("draft_sections",  "assemble_report")
workflow.add_edge("assemble_report", "finalize_report")
workflow.add_edge("finalize_report", END)

report_generation_agent = workflow.compile()
