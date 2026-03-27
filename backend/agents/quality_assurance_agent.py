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

# Primary LLM (existing)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
)

# Secondary LLM (pluggable — currently Groq, easy to replace later)
llm_secondary = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
)

MIN_WORDS_FOR_PASS  = 100
MIN_QUALITY_SCORE   = 5.0
MIN_CITATION_COUNT  = 1
MAX_ALLOWED_DISAGREEMENT = 3.0  # NEW

# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class QAState(TypedDict):
    synthesis_draft:    str
    extracted_data:     Dict
    research_questions: List[str]

    coverage_result:    Dict
    citation_result:    Dict

    quality_report:     Dict
    quality_passed:     bool


# -------------------------------------------------------------------
# OUTPUT MODELS
# -------------------------------------------------------------------

class CoverageResult(BaseModel):
    covered_questions:  List[str]
    missing_questions:  List[str]
    coverage_score:     float


class MultiLLMQualityScore(BaseModel):
    score: float
    confidence: float
    strengths: List[str]
    weaknesses: List[str]
    reasoning: str
    suggestions: List[str]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _count_inline_citations(text: str) -> int:
    return len(re.findall(r'\[[^\]]{3,80}\]', text))


# -------------------------------------------------------------------
# NODE 1 — VALIDATE COVERAGE (UNCHANGED)
# -------------------------------------------------------------------

def validate_coverage_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 1: VALIDATING RESEARCH QUESTION COVERAGE")
    print("=" * 60)

    draft  = state.get("synthesis_draft", "")
    rqs    = state.get("research_questions", [])

    if not draft:
        return {"coverage_result": {
            "covered_questions": [],
            "missing_questions": rqs,
            "coverage_score": 0.0
        }}

    rq_text = "\n".join(f"- {q}" for q in rqs)
    parser  = PydanticOutputParser(pydantic_object=CoverageResult)

    prompt = f"""Evaluate which research questions are clearly addressed.

Research Questions:
{rq_text}

Draft:
{draft[:4000]}

Output JSON only.

{parser.get_format_instructions()}
"""
    try:
        response = llm.invoke(prompt)
        parsed   = parser.parse(response.content)
        result   = parsed.model_dump()
    except Exception:
        half = len(rqs) // 2
        result = {
            "covered_questions": rqs[:half],
            "missing_questions": rqs[half:],
            "coverage_score": half / len(rqs) if rqs else 0.0
        }

    print(f"Coverage: {result['coverage_score']:.2f}")
    return {"coverage_result": result}


# -------------------------------------------------------------------
# NODE 2 — CHECK CITATIONS (UNCHANGED)
# -------------------------------------------------------------------

def check_citations_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 2: CHECKING INLINE CITATIONS")
    print("=" * 60)

    draft  = state.get("synthesis_draft", "")
    papers = state.get("extracted_data", {}).get("papers", [])

    citation_count = _count_inline_citations(draft)

    return {"citation_result": {
        "citation_count": citation_count,
        "total_papers_available": len(papers),
        "has_citations": citation_count >= MIN_CITATION_COUNT,
        "citation_density": citation_count
    }}


# -------------------------------------------------------------------
# NODE 3 — MULTI-LLM SCORING
# -------------------------------------------------------------------

def score_quality_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 3: MULTI-LLM QUALITY SCORING")
    print("=" * 60)

    draft = state.get("synthesis_draft", "")
    rqs   = state.get("research_questions", [])
    coverage_result  = state.get("coverage_result", {})
    citation_result  = state.get("citation_result", {})

    word_count = len(draft.split())
    rq_text    = "\n".join(f"- {q}" for q in rqs)

    parser = PydanticOutputParser(pydantic_object=MultiLLMQualityScore)

    prompt = f"""You are an expert reviewer.

Research Questions:
{rq_text}

Metrics:
- Word count: {word_count}
- Coverage: {coverage_result.get("coverage_score", 0)}
- Citations: {citation_result.get("citation_count", 0)}

Draft:
{draft[:3000]}

CRITICAL:
- JSON only
- score: 0-10
- confidence: 0-1
- strengths: 2-5
- weaknesses: 2-5

{parser.get_format_instructions()}
"""

    results = []
    models = [("Primary", llm), ("Secondary", llm_secondary)]

    for name, model in models:
        try:
            response = model.invoke(prompt)
            parsed = parser.parse(response.content)
            data = parsed.model_dump()

            print(f"{name} Score: {data['score']} | Confidence: {data['confidence']}")
            results.append(data)

        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    if not results:
        return {"quality_report": {"score": 0.0, "reasoning": "All models failed"}}

    # ---------------- AGGREGATION ----------------

    scores = [r["score"] for r in results]
    confidences = [r["confidence"] for r in results]

    # Confidence-weighted average
    total_conf = sum(confidences)
    if total_conf == 0:
        final_score = sum(scores) / len(scores)
    else:
        final_score = sum(s * c for s, c in zip(scores, confidences)) / total_conf

    disagreement = max(scores) - min(scores)

    print(f"Disagreement: {disagreement:.2f}")

    # Merge feedback
    strengths = list(set(s for r in results for s in r["strengths"]))
    weaknesses = list(set(w for r in results for w in r["weaknesses"]))
    suggestions = list(set(s for r in results for s in r["suggestions"]))

    return {
        "quality_report": {
            "score": round(final_score, 1),
            "confidence": round(sum(confidences)/len(confidences), 2),
            "disagreement": round(disagreement, 2),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "reasoning": "Multi-LLM evaluation"
        }
    }


# -------------------------------------------------------------------
# NODE 4 — FINALIZE QA
# -------------------------------------------------------------------

def finalize_qa_node(state: QAState) -> Dict:
    print("\n" + "=" * 60)
    print("QA STAGE 4: FINALIZING")
    print("=" * 60)

    qr = state.get("quality_report", {})
    coverage_result = state.get("coverage_result", {})
    citation_result = state.get("citation_result", {})

    score = qr.get("score", 0)
    disagreement = qr.get("disagreement", 0)
    word_count = len(state.get("synthesis_draft", "").split())

    passed = (
        word_count >= MIN_WORDS_FOR_PASS and
        score >= MIN_QUALITY_SCORE and
        citation_result.get("has_citations", False) and
        disagreement < MAX_ALLOWED_DISAGREEMENT
    )

    final_report = {
        **qr,
        "coverage_score":     coverage_result.get("coverage_score", 0),
        "covered_questions":  coverage_result.get("covered_questions", []),
        "missing_questions":  coverage_result.get("missing_questions", []),
        "citation_count":     citation_result.get("citation_count", 0),
        "word_count":         word_count,
        "passed":             passed,
        "status":             "passed" if passed else "needs_improvement",
        "summary":            f"Score {score}/10 | Disagreement {disagreement}"
    }

    print(f"Final Score: {score} | Passed: {passed}")

    return {
        "quality_report": final_report,
        "quality_passed": passed
    }


# -------------------------------------------------------------------
# GRAPH
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