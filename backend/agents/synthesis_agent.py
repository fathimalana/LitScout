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
    temperature=0.2,   # slight creativity for prose generation
)

MAX_CHARS_PER_PAPER   = 1500   # matches thematic agent
THEME_BATCH_SIZE      = 3      # themes per LLM call (stay within rate limit)
MAX_PAPERS_FOR_CONTEXT = 20    # how many papers to surface as supporting evidence


# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class SynthesisState(TypedDict):
    # Inputs (from orchestrator)
    themes: List[Dict]
    extracted_data: Dict
    research_questions: List[str]
    user_prompt: str
    synthesis_style: str          # e.g. "academic_literature_review"

    # Intermediate
    context_blocks: List[str]     # per-theme context strings built in stage 1
    section_drafts: List[str]     # per-theme prose written in stage 2

    # Outputs
    synthesis_draft: str
    synthesis_results: Dict


# -------------------------------------------------------------------
# PYDANTIC OUTPUT MODELS
# -------------------------------------------------------------------

class SectionDraft(BaseModel):
    theme_name: str
    section_text: str


class SectionDraftList(BaseModel):
    sections: List[SectionDraft]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _truncate(text: str, limit: int = MAX_CHARS_PER_PAPER) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _paper_snippet(paper: Dict) -> str:
    """Return a compact, citation-keyed snippet for a paper."""
    title  = paper.get("title", "Untitled")
    year   = paper.get("year", "")

    # Generate a readable citation key
    authors = paper.get("authors", [])
    author_text = "Unknown"
    if authors and isinstance(authors, list):
        if isinstance(authors[0], dict) and "name" in authors[0]:
            names = [a.get("name", "Unknown") for a in authors]
        else:
            names = [str(a) for a in authors]
            
        if len(names) == 1:
            author_text = names[0].split()[-1]
        elif len(names) == 2:
            author_text = f"{names[0].split()[-1]} and {names[1].split()[-1]}"
        else:
            author_text = f"{names[0].split()[-1]} et al."
            
    citation_key = f"{author_text}, {year}" if year else author_text

    # Prefer extracted sections; fall back to abstract
    content = ""
    sections = paper.get("extracted_text", {})
    if isinstance(sections, dict):
        for k in ["results", "discussion", "conclusion", "abstract"]:
            if sections.get(k):
                content += sections[k] + " "
    if not content:
        content = paper.get("abstract", "")

    return f"[{citation_key}] {title}\n{_truncate(content.strip())}"


def _batch(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# -------------------------------------------------------------------
# NODE 1 — PREPARE CONTEXT
# -------------------------------------------------------------------

def prepare_context_node(state: SynthesisState) -> Dict:
    print("\n" + "=" * 60)
    print("SYNTHESIS STAGE 1: PREPARING CONTEXT")
    print("=" * 60)

    themes         = state.get("themes", [])
    extracted_data = state.get("extracted_data", {})
    papers         = extracted_data.get("papers", [])

    if not themes:
        print("⚠️  No themes found — synthesis will be minimal.")
        return {"context_blocks": []}

    # Build a lookup: paperId → snippet
    paper_lookup: Dict[str, str] = {}
    for paper in papers[:MAX_PAPERS_FOR_CONTEXT]:
        pid = paper.get("paperId") or paper.get("title", "")[:40]
        if pid:
            paper_lookup[pid] = _paper_snippet(paper)

    context_blocks: List[str] = []
    for theme in themes:
        name        = theme.get("name", "Untitled Theme")
        description = theme.get("description", "")
        findings    = theme.get("findings", [])
        citations   = theme.get("citations", [])

        # Gather supporting paper snippets for this theme
        supporting = []
        for cid in citations[:5]:
            snippet = paper_lookup.get(cid)
            if not snippet:
                # Fuzzy fallback: search by title fragment
                for pid, snip in paper_lookup.items():
                    if cid.lower() in pid.lower() or pid.lower() in cid.lower():
                        snippet = snip
                        break
            if snippet:
                supporting.append(snippet)

        block = (
            f"THEME: {name}\n"
            f"DESCRIPTION: {description}\n"
            f"KEY FINDINGS:\n" + "\n".join(f"  - {f}" for f in findings) + "\n"
            f"SUPPORTING PAPERS ({len(supporting)} of {len(citations)}):\n"
            + "\n\n".join(supporting[:3])
        )
        context_blocks.append(block)

    print(f"Prepared context blocks for {len(context_blocks)} themes.")
    return {"context_blocks": context_blocks}


# -------------------------------------------------------------------
# NODE 2 — DRAFT SECTIONS (per-theme, batched)
# -------------------------------------------------------------------

def draft_sections_node(state: SynthesisState) -> Dict:
    print("\n" + "=" * 60)
    print("SYNTHESIS STAGE 2: DRAFTING THEME SECTIONS")
    print("=" * 60)

    themes         = state.get("themes", [])
    context_blocks = state.get("context_blocks", [])
    research_questions = state.get("research_questions", [])

    if not themes or not context_blocks:
        print("⚠️  Skipping section drafting — no themes or context.")
        return {"section_drafts": []}

    rq_text  = "\n".join(f"- {q}" for q in research_questions)
    parser   = PydanticOutputParser(pydantic_object=SectionDraftList)
    all_drafts: List[str] = []

    theme_context_pairs = list(zip(themes, context_blocks))

    for batch_idx, batch in enumerate(_batch(theme_context_pairs, THEME_BATCH_SIZE)):
        print(f"  Drafting batch {batch_idx + 1} ({len(batch)} themes)…")

        batch_text = "\n\n---\n\n".join(
            f"CONTEXT BLOCK {i + 1}:\n{ctx}"
            for i, (_, ctx) in enumerate(batch)
        )

        theme_names = [t.get("name", f"Theme {i+1}") for i, (t, _) in enumerate(batch)]

        prompt = f"""You are an expert academic writer conducting a systematic literature review.

Research Questions:
{rq_text}

Write a 2–3 paragraph synthesis section for EACH of the following themes.
Each section must:
1. Synthesize the evidence from the supporting papers. YOU MUST use the EXACT citation keys provided in brackets, e.g., [Smith et al., 2023]. Do not use parentheses or omit brackets.
2. Highlight agreements, contradictions, or gaps across the papers
3. Use formal academic prose appropriate for a journal literature review
4. Stay focused on the theme — do NOT introduce unrelated content

Themes to write about: {json.dumps(theme_names)}

Context for each theme:
{batch_text}

CRITICAL:
- Output MUST be valid JSON only.
- Do NOT include markdown formatting inside section_text (no **, no #, etc.)
- Do NOT add explanatory text outside the JSON.
- EVERY single citation MUST be enclosed in square brackets exactly as provided in the context, e.g. [Smith et al., 2023]. Do NOT use parentheses like (Smith, 2023) or plain text citations. This is REQUIRED for quality assurance checks.

{parser.get_format_instructions()}
"""
        try:
            response = llm.invoke(prompt)
            parsed   = parser.parse(response.content)
            for section in parsed.sections:
                all_drafts.append(section.section_text)
        except Exception as e:
            print(f"  ⚠️  Batch {batch_idx + 1} parsing failed: {e}")
            # Fallback: raw content as plain text
            all_drafts.append(response.content[:2000] if 'response' in dir() else "")

    print(f"Drafted {len(all_drafts)} theme sections.")
    return {"section_drafts": all_drafts}


# -------------------------------------------------------------------
# NODE 3 — INTEGRATE DRAFT (stitch into full narrative)
# -------------------------------------------------------------------

def integrate_draft_node(state: SynthesisState) -> Dict:
    print("\n" + "=" * 60)
    print("SYNTHESIS STAGE 3: INTEGRATING INTO FULL NARRATIVE")
    print("=" * 60)

    themes          = state.get("themes", [])
    section_drafts  = state.get("section_drafts", [])
    research_questions = state.get("research_questions", [])
    user_prompt     = state.get("user_prompt", "")
    synthesis_style = state.get("synthesis_style", "academic_literature_review")

    if not section_drafts:
        fallback = (
            "Synthesis could not be completed: no theme sections were generated. "
            "This may be due to insufficient extracted paper content."
        )
        print("⚠️  No section drafts — returning fallback synthesis.")
        return {"synthesis_draft": fallback}

    rq_text = "\n".join(f"- {q}" for q in research_questions)

    # Pair section drafts with theme names
    sections_block = ""
    for i, (theme, draft) in enumerate(zip(themes, section_drafts), 1):
        theme_name = theme.get("name", f"Theme {i}")
        sections_block += f"\n\n## {i}. {theme_name}\n{draft}"

    prompt = f"""You are an expert academic writer finalising a systematic literature review.

Original Research Topic: {user_prompt}

Research Questions:
{rq_text}

Style: {synthesis_style.replace("_", " ").title()}

You have been provided with individual theme sections (written earlier). Your task is to:
1. Write a 2–3 sentence INTRODUCTION paragraph that contextualises the review topic and states its scope
2. Insert each theme section (do NOT rewrite them — copy them exactly as provided in order to preserve the [Citation] brackets)  
3. Write a 3–4 sentence CONCLUSION paragraph that summarises cross-cutting patterns, key tensions, and future research directions
4. Do NOT add references section (citations are inline already) and ensure any new citations maintain the [Citation] format
5. Use plain academic prose — no markdown formatting characters

THEME SECTIONS (in order):
{sections_block}

Output the complete, integrated synthesis text:"""

    try:
        response = llm.invoke(prompt)
        synthesis_draft = response.content.strip()
        print(f"Integrated synthesis: {len(synthesis_draft.split())} words.")
    except Exception as e:
        print(f"⚠️  Integration failed: {e}. Using concatenated sections as fallback.")
        synthesis_draft = "\n\n".join(section_drafts)

    return {"synthesis_draft": synthesis_draft}


# -------------------------------------------------------------------
# NODE 4 — FINALIZE
# -------------------------------------------------------------------

def finalize_node(state: SynthesisState) -> Dict:
    print("\n" + "=" * 60)
    print("SYNTHESIS STAGE 4: FINALIZING")
    print("=" * 60)

    draft  = state.get("synthesis_draft", "")
    themes = state.get("themes", [])

    word_count = len(draft.split())

    synthesis_results = {
        "total_themes":       len(themes),
        "word_count":         word_count,
        "section_count":      len(state.get("section_drafts", [])),
        "synthesis_style":    state.get("synthesis_style", "academic_literature_review"),
        "method":             "LLM batch-drafting with integrated narrative (LangGraph)",
        "status":             "complete" if word_count > 50 else "partial"
    }

    print(f"Synthesis complete: {word_count} words, {len(themes)} themes covered.")
    return {
        "synthesis_draft":   draft,
        "synthesis_results": synthesis_results
    }


# -------------------------------------------------------------------
# GRAPH ASSEMBLY
# -------------------------------------------------------------------

workflow = StateGraph(SynthesisState)

workflow.add_node("prepare_context", prepare_context_node)
workflow.add_node("draft_sections",  draft_sections_node)
workflow.add_node("integrate_draft", integrate_draft_node)
workflow.add_node("finalize",        finalize_node)

workflow.set_entry_point("prepare_context")
workflow.add_edge("prepare_context", "draft_sections")
workflow.add_edge("draft_sections",  "integrate_draft")
workflow.add_edge("integrate_draft", "finalize")
workflow.add_edge("finalize",        END)

synthesis_agent = workflow.compile()