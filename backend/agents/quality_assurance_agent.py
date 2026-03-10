import os
import re
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
    temperature=0,
)

MIN_WORDS_FOR_PASS  = 100   # minimum synthesis draft length
MIN_QUALITY_SCORE   = 5.0   # LLM score threshold out of 10
MIN_CITATION_COUNT  = 1     # at least 1 inline citation required


# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class QAState(TypedDict):
    # Inputs (from orchestrator)
    synthesis_draft:    str
    extracted_data:     Dict
    research_questions: List[str]

    # Intermediate
    coverage_result:    Dict   # which RQs are addressed / missing
    citation_result:    Dict   # citation count + presence flag

    # Outputs
    quality_report:     Dict
    quality_passed:     bool


# -------------------------------------------------------------------
# PYDANTIC OUTPUT MODELS
# -------------------------------------------------------------------

class CoverageResult(BaseModel):
    covered_questions:  List[str] = Field(description="Research questions clearly addressed in the draft.")
    missing_questions:  List[str] = Field(description="Research questions not addressed or only superficially touched.")
    coverage_score:     float     = Field(description="Coverage ratio from 0.0 to 1.0.")

class QualityScore(BaseModel):
    score:       float       = Field(description="Overall quality score from 0.0 to 10.0.")
    reasoning:   str         = Field(description="One-paragraph explanation of the score.")
    suggestions: List[str]   = Field(description="Concrete improvement suggestions (up to 5).")


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _count_inline_citations(text: str) -> int:
    """Count occurrences of [paper_id] style citations in the draft."""
    return len(re.findall(r'\[[^\]]{3,80}\]', text))


# -------------------------------------------------------------------
# NODE 1 — VALIDATE COVERAGE
# -------------------------------------------------------------------

def validate_coverage_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 1: VALIDATING RESEARCH QUESTION COVERAGE")
    print("=" * 60)

    draft  = state.get("synthesis_draft", "")
    rqs    = state.get("research_questions", [])

    if not draft:
        print("⚠️  Empty synthesis draft — skipping coverage check.")
        return {"coverage_result": {
            "covered_questions": [],
            "missing_questions": rqs,
            "coverage_score": 0.0
        }}

    if not rqs:
        print("⚠️  No research questions provided — skipping coverage check.")
        return {"coverage_result": {
            "covered_questions": [],
            "missing_questions": [],
            "coverage_score": 1.0
        }}

    rq_text = "\n".join(f"- {q}" for q in rqs)
    parser  = PydanticOutputParser(pydantic_object=CoverageResult)

    prompt = f"""You are a systematic literature review quality assessor.

Research Questions:
{rq_text}

Literature Review Draft:
{draft[:4000]}

Task: Determine which research questions are CLEARLY ADDRESSED in the draft.
A question is "covered" only if the draft contains substantive discussion relevant to it (not just a mention).

CRITICAL:
- Output MUST be valid JSON only.
- Do NOT add markdown or prose outside the JSON.

{parser.get_format_instructions()}
"""
    try:
        response = llm.invoke(prompt)
        parsed   = parser.parse(response.content)
        result   = parsed.model_dump()
    except Exception as e:
        print(f"  ⚠️  Coverage parsing failed: {e}")
        # Fallback: assume partial coverage
        half = len(rqs) // 2
        result = {
            "covered_questions":  rqs[:half],
            "missing_questions":  rqs[half:],
            "coverage_score":     half / len(rqs) if rqs else 0.0
        }

    print(f"Coverage: {len(result['covered_questions'])}/{len(rqs)} RQs addressed "
          f"(score: {result['coverage_score']:.2f})")
    return {"coverage_result": result}


# -------------------------------------------------------------------
# NODE 2 — CHECK CITATIONS
# -------------------------------------------------------------------

def check_citations_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 2: CHECKING INLINE CITATIONS")
    print("=" * 60)

    draft        = state.get("synthesis_draft", "")
    papers       = state.get("extracted_data", {}).get("papers", [])
    total_papers = len(papers)

    citation_count = _count_inline_citations(draft)
    has_citations  = citation_count >= MIN_CITATION_COUNT

    result = {
        "citation_count":        citation_count,
        "total_papers_available": total_papers,
        "has_citations":         has_citations,
        "citation_density":      round(citation_count / max(len(draft.split()), 1) * 100, 2)
    }

    status = "✅" if has_citations else "⚠️"
    print(f"{status} Found {citation_count} inline citations across {len(draft.split())} words.")
    return {"citation_result": result}


# -------------------------------------------------------------------
# NODE 3 — SCORE QUALITY
# -------------------------------------------------------------------

def score_quality_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 3: LLM QUALITY SCORING")
    print("=" * 60)

    draft            = state.get("synthesis_draft", "")
    rqs              = state.get("research_questions", [])
    coverage_result  = state.get("coverage_result", {})
    citation_result  = state.get("citation_result", {})

    word_count       = len(draft.split())
    rq_text          = "\n".join(f"- {q}" for q in rqs)
    coverage_score   = coverage_result.get("coverage_score", 0.0)
    citation_count   = citation_result.get("citation_count", 0)

    parser = PydanticOutputParser(pydantic_object=QualityScore)

    prompt = f"""You are an expert peer-reviewer evaluating a systematic literature review draft.

Research Questions:
{rq_text}

Pre-computed Metrics:
- Word count: {word_count}
- RQ Coverage Score: {coverage_score:.2f} / 1.0
- Inline citations found: {citation_count}
- Missing RQs: {coverage_result.get("missing_questions", [])}

Draft (first 3000 words):
{draft[:3000]}

Evaluate the draft on these criteria:
1. Academic writing quality and coherence
2. Evidence-based argumentation
3. Synthesis depth (not just summary)
4. Logical flow and structure
5. Balance between themes

Provide:
- score: 0.0–10.0 (consider pre-computed metrics)
- reasoning: concise explanation
- suggestions: up to 5 specific, actionable improvements

CRITICAL: Output MUST be valid JSON only. No markdown outside JSON.

{parser.get_format_instructions()}
"""
    try:
        response = llm.invoke(prompt)
        parsed   = parser.parse(response.content)
        result   = parsed.model_dump()
    except Exception as e:
        print(f"  ⚠️  Quality scoring failed: {e}")
        # Derive a basic score from pre-computed metrics
        score = round(min(10.0, (coverage_score * 4) + (min(citation_count, 5) * 0.5) + (min(word_count, 500) / 500 * 3)), 1)
        result = {
            "score":       score,
            "reasoning":   "Scoring based on coverage and citation metrics (LLM scoring unavailable).",
            "suggestions": ["Ensure all research questions are addressed.", "Add more inline citations."]
        }

    print(f"Quality score: {result['score']:.1f}/10.0")
    return {"quality_report": result}


# -------------------------------------------------------------------
# NODE 4 — FINALIZE QA
# -------------------------------------------------------------------

def finalize_qa_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 4: FINALIZING QA REPORT")
    print("=" * 60)

    draft           = state.get("synthesis_draft", "")
    quality_score   = state.get("quality_report", {})
    coverage_result = state.get("coverage_result", {})
    citation_result = state.get("citation_result", {})
    rqs             = state.get("research_questions", [])

    word_count      = len(draft.split())
    score           = quality_score.get("score", 0.0)

    # Determine pass/fail
    passed = (
        word_count         >= MIN_WORDS_FOR_PASS  and
        score              >= MIN_QUALITY_SCORE    and
        citation_result.get("has_citations", False)
    )

    final_report = {
        # Scores
        "score":                    score,
        "coverage_score":           coverage_result.get("coverage_score", 0.0),
        "citation_count":           citation_result.get("citation_count", 0),
        "word_count":               word_count,

        # Detailed results
        "covered_questions":        coverage_result.get("covered_questions", []),
        "missing_questions":        coverage_result.get("missing_questions", []),
        "has_citations":            citation_result.get("has_citations", False),
        "citation_density":         citation_result.get("citation_density", 0.0),
        "total_papers_available":   citation_result.get("total_papers_available", 0),

        # LLM assessment
        "reasoning":                quality_score.get("reasoning", ""),
        "suggestions":              quality_score.get("suggestions", []),

        # Summary
        "summary":                  f"Quality score {score:.1f}/10. "
                                    f"Coverage: {coverage_result.get('coverage_score', 0.0):.0%} of RQs. "
                                    f"{citation_result.get('citation_count', 0)} inline citations found.",
        "passed":                   passed,
        "method":                   "LangGraph multi-check QA pipeline",
        "status":                   "passed" if passed else "needs_improvement"
    }

    print(f"QA {'✅ PASSED' if passed else '⚠️  NEEDS IMPROVEMENT'}: "
          f"score={score:.1f}, words={word_count}, citations={citation_result.get('citation_count',0)}, "
          f"coverage={coverage_result.get('coverage_score', 0.0):.0%}")

    return {
        "quality_report": final_report,
        "quality_passed": passed
    }


# -------------------------------------------------------------------
# GRAPH ASSEMBLY
# -------------------------------------------------------------------

workflow = StateGraph(QAState)

workflow.add_node("validate_coverage", validate_coverage_node)
workflow.add_node("check_citations",   check_citations_node)
workflow.add_node("score_quality",     score_quality_node)
workflow.add_node("finalize_qa",       finalize_qa_node)

workflow.set_entry_point("validate_coverage")
workflow.add_edge("validate_coverage", "check_citations")
workflow.add_edge("check_citations",   "score_quality")
workflow.add_edge("score_quality",     "finalize_qa")
workflow.add_edge("finalize_qa",       END)

qa_agent = workflow.compile()
