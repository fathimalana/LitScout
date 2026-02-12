import json
import sys
import os
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
if env_path:
    load_dotenv(env_path, override=True)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__)) # Add backend directory to path

from pydantic import BaseModel, Field
from langchain_core.tools import tool, render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- API KEY VALIDATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY :
    raise ValueError("ERROR: Neither GROQ_API_KEY nor GOOGLE_API_KEY found in environment")

# Initialize LLM (prefer Groq if available)
if GROQ_API_KEY:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )


# --- MOCK AGENTS FOR DEVELOPMENT ---
class MockAgent:
    def invoke(self, input_data):
        print(f"MockAgent invoked with keys: {list(input_data.keys())}")
        return {"status": "mock_response", "data": input_data}

# Default to mocks
screening_agent = MockAgent()
extraction_agent = MockAgent()
thematic_agent = MockAgent()
synthesis_agent = MockAgent()
qa_agent = MockAgent()

# Import real agents if available
try:
    from agents.search_and_filter_agent import saf_agent, REPUTED_SOURCES
except ImportError as e:
    saf_agent = MockAgent()
    REPUTED_SOURCES = []

try:
    from agents.screening_agent import screening_agent as real_screening
    screening_agent = real_screening
except ImportError as e:
    print(f"Using mock Screening agent: {e}")

try:
    from agents.extraction_agent import extraction_agent as real_extraction
    extraction_agent = real_extraction
except ImportError as e:
    print(f"Using mock Extraction agent: {e}")

try:
    from agents.thematic_analysis_agent import thematic_analysis_agent as real_thematic
    thematic_agent = real_thematic
except ImportError as e:
    print(f"Using mock Thematic Analysis agent: {e}")

# Import adaptive extraction module
try:
    from agents.adaptive_extraction import adaptive_extraction_loop
    USE_ADAPTIVE_EXTRACTION = True
except ImportError as e:
    print(f"Adaptive extraction not available: {e}")
    USE_ADAPTIVE_EXTRACTION = False

# --- 1. STATE DEFINITION ---
class LitScoutState(BaseModel):
    """Complete state management for LitScout"""
    
    # Input
    user_prompt: str = ""
    
    # Planning
    plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    next_agent: str = ""
    observations: List[str] = Field(default_factory=list)
    
    # Research Planning Phase
    research_questions: List[str] = Field(default_factory=list)
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    
    # Search & Filter Phase
    search_query: str = ""
    raw_papers: List[Dict] = Field(default_factory=list)
    filtered_papers: List[Dict] = Field(default_factory=list)
    
    # Screening Phase
    screened_papers: List[Dict] = Field(default_factory=list)
    screening_results: Dict = Field(default_factory=dict)
    
    # Extraction Phase
    extracted_data: Dict = Field(default_factory=dict)
    extraction_results: Dict = Field(default_factory=dict)
    
    # Thematic Analysis Phase
    themes: List[Dict] = Field(default_factory=list)
    thematic_results: Dict = Field(default_factory=dict)
    
    # Synthesis Phase
    synthesis_draft: str = ""
    synthesis_results: Dict = Field(default_factory=dict)
    
    # Quality Assurance Phase
    quality_report: Dict = Field(default_factory=dict)
    quality_passed: bool = False
    
    # Final Output
    final_report: str = ""
    
    # Workflow Control
    workflow_complete: bool = False
    error_messages: List[str] = Field(default_factory=list)

# --- 2. PARSER SETUP ---
class PlannerResponse(BaseModel):
    """The JSON response structure for the planner's decision."""
    plan: List[str] = Field(description="The updated step-by-step plan.")
    next_agent: str = Field(description="The name of the tool to call next, or 'END'.")

parser = PydanticOutputParser(pydantic_object=PlannerResponse)

# --- 3. AGENT TOOL DEFINITIONS ---
class ResearchPlan(BaseModel):
    research_questions: List[str] = Field(description="List of 3-5 specific research questions.")
    inclusion_criteria: str = Field(description="Concise inclusion criteria for the literature review.")
    exclusion_criteria: str = Field(description="Concise exclusion criteria for the literature review.")

@tool
def formulate_research_plan(user_prompt: str) -> Dict[str, Any]:
    """
    Formulates a research plan based on the user_prompt.
    Generates research questions and defines inclusion/exclusion criteria using an LLM.
    """
    print("---TOOL: Formulating Research Plan---")
    
    try:
        plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        prompt_template_string = """Generate a research plan for a literature review.

Research Prompt: {user_prompt}

Generate:
1. **research_questions**: 3-5 specific research questions
2. **inclusion_criteria**: CONTENT-FOCUSED keywords that papers should contain
3. **exclusion_criteria**: CONTENT to AVOID (methods/topics that are out of scope)

{format_instructions}
"""
        
        prompt = ChatPromptTemplate.from_template(
            prompt_template_string,
            partial_variables={"format_instructions": plan_parser.get_format_instructions()},
        )
        
        chain = prompt | llm | plan_parser
        response_obj = chain.invoke({"user_prompt": user_prompt})
        
        result = response_obj.model_dump()
        
        print(f"\nGenerated {len(result['research_questions'])} research questions:")
        for i, q in enumerate(result['research_questions'], 1):
            print(f"   {i}. {q}")
        print(f"\nInclusion: {result['inclusion_criteria']}")
        print(f"Exclusion: {result['exclusion_criteria']}\n")
        
        return result
        
    except Exception as e:
        print(f"Error in formulate_research_plan: {e}")
        return {
            "research_questions": [f"Research question related to: {user_prompt}"],
            "inclusion_criteria": "Relevant academic papers published in the last 10 years",
            "exclusion_criteria": "Non-peer-reviewed sources, duplicate studies",
            "error": str(e)
        }

@tool
def search_and_filter_tool(research_questions: List[str], inclusion_criteria: str, 
                           exclusion_criteria: str, user_prompt: str) -> Dict[str, Any]:
    """
    Search and Filter Agent Tool - Delegates to the specialized SAF agent.
    """
    print("---TOOL: Search and Filter Agent---")
    
    try:
        import datetime
        current_year = datetime.datetime.now().year
        
        # Extract metadata using LLM
        extraction_prompt = f"""Extract filtering criteria from this research query:

Current Year: {current_year}

"{user_prompt}"

Respond with ONLY a JSON object:
{{
  "start_year": <number or null>,
  "end_year": <number or null>,
  "sources": [<list of venues or empty array>]
}}

If no specific years mentioned, use null for both (defaults to last 5 years from current year).
Consider the current year when interpreting relative time references like "recent", "last decade", etc.
"""
        
        try:
            response = llm.invoke(extraction_prompt)
            content = response.content.strip().replace("```json", "").replace("```", "").strip()
            metadata = json.loads(content)
            
            start_year = metadata.get("start_year") or (current_year - 5)
            end_year = metadata.get("end_year") or current_year
            sources = metadata.get("sources", [])
            
        except Exception as e:
            print(f"Could not extract metadata: {e}, using defaults")
            start_year = current_year - 5
            end_year = current_year
            sources = []
        
        # Merge with REPUTED_SOURCES if sources list is empty or generic
        if not sources:
            sources = REPUTED_SOURCES
        
        saf_input = {
            "research_questions": research_questions,
            "start_year": start_year,
            "end_year": end_year,
            "sources": sources,
            "raw_papers": [],
            "filtered_papers": []
        }
        
        print(f"Year range: {start_year}-{end_year}")
        print(f"Sources: {sources if sources else 'All venues'}")
        
        saf_result = saf_agent.invoke(saf_input)
        
        papers_found = len(saf_result.get('filtered_papers', []))
        print(f"SAF completed: {papers_found} papers found")
        
        return saf_result
        
    except Exception as e:
        print(f"Error in search_and_filter_tool: {e}")
        return {
            "raw_papers": [],
            "filtered_papers": [],
            "error": str(e)
        }

@tool
def screening_tool(filtered_papers: List[Dict], research_questions: List[str], 
                   inclusion_criteria: str, exclusion_criteria: str) -> Dict[str, Any]:
    """
    Screening Agent Tool - Content-based relevance screening.
    """
    print("---TOOL: Screening Agent---")
    
    try:
        screening_input = {
            "filtered_papers": filtered_papers,
            "research_questions": research_questions,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "keyword_high_threshold": 0.0,
            "keyword_medium_threshold": 0.0,
            "tfidf_threshold": 0.0,
            "use_llm_screening": False,
            "year_range": (1900, 2100),
            "allowed_publication_types": [],
            "min_citation_count": 0,
            "metadata_filtered_papers": [],
            "high_relevance_papers": [],
            "medium_relevance_papers": [],
            "borderline_papers": [],
            "screened_papers": [],
            "screening_results": {}
        }
        
        print(f"Screening {len(filtered_papers)} papers")
        screening_result = screening_agent.invoke(screening_input)
        
        papers_screened = len(screening_result.get('screened_papers', []))
        print(f"Screening completed: {papers_screened} papers passed")
        
        return screening_result
        
    except Exception as e:
        print(f"Error in screening_tool: {e}")
        return {
            "screened_papers": filtered_papers[:100] if filtered_papers else [],
            "screening_results": {"error": str(e)},
            "error": str(e)
        }

@tool
def extraction_tool(screened_papers: List[Dict], research_questions: List[str] = None) -> Dict[str, Any]:
    """Extraction Agent Tool - Fetches PDFs and extracts text using PyMuPDF with adaptive limiting"""
    print("---TOOL: Extraction Agent---")
    try:
        # Use adaptive extraction if available
        if USE_ADAPTIVE_EXTRACTION:
            print("\n🤖 Using ADAPTIVE EXTRACTION with LLM-driven limit adjustment")
            
            # Use research questions from parameter or fallback
            if not research_questions:
                research_questions = ["Literature review on the given topic"]
            
            # Run adaptive extraction loop
            extracted_papers, metadata = adaptive_extraction_loop(
                screened_papers=screened_papers,
                research_questions=research_questions,
                extraction_agent=extraction_agent
            )
            
            print(f"\n✅ Adaptive extraction completed:")
            print(f"   Total iterations: {len(metadata['iterations'])}")
            print(f"   Papers extracted: {metadata['total_extracted']}")
            print(f"   Success rate: {metadata['total_extracted']/metadata['total_attempted']:.1%}")
            
            return {
                "extracted_data": {"papers": extracted_papers},
                "extraction_results": metadata
            }
        else:
            # Fallback to original extraction (all papers)
            print("\n⚠️ Adaptive extraction not available, extracting all papers")
            extraction_input = {
                "screened_papers": screened_papers,
                "papers_with_pdfs": [],
                "papers_with_text": [],
                "extraction_results": {}
            }
            print(f"Extracting from {len(screened_papers)} papers")
            extraction_result = extraction_agent.invoke(extraction_input)
            
            # Get results from new format
            papers_extracted = len(extraction_result.get('papers_with_text', []))
            print(f"Extraction completed: {papers_extracted} papers extracted")
            
            # Convert to format expected by orchestrator
            return {
                "extracted_data": {"papers": extraction_result.get('papers_with_text', [])},
                "extraction_results": extraction_result.get('extraction_results', {})
            }
    except Exception as e:
        print(f"Error in extraction_tool: {e}")
        import traceback
        traceback.print_exc()
        return {"extracted_data": {}, "extraction_results": {}, "error": str(e)}

@tool
def thematic_analysis_tool(extracted_data: Dict, research_questions: List[str]) -> Dict[str, Any]:
    """Thematic Analysis Agent Tool"""
    print("---TOOL: Thematic Analysis Agent---")
    try:
        thematic_input = {
            "extracted_data": extracted_data,
            "research_questions": research_questions,
            "aggregated_text": [],
            "initial_themes": [],
            "themes": [],
            "thematic_results": {}
        }

        print(f"Running thematic analysis")
        thematic_result = thematic_agent.invoke(thematic_input)

        themes = thematic_result.get("themes", [])
        print(f"Thematic analysis completed: {len(themes)} themes identified")
        for i, theme in enumerate(themes, 1):
            print(f"\nTHEME {i}: {theme.get('name', 'Untitled')}")
            print(f"Description: {theme.get('description', 'No description')}")
            print("Findings:")
            for finding in theme.get("findings", [])[:3]:
                print(f"  - {finding}")
            print(f"Supported by {len(theme.get('citations', []))} papers")
        return {
            "themes": themes,
            "thematic_results": thematic_result.get("thematic_results", {})
        }

    except Exception as e:
        print(f"Error in thematic_analysis_tool: {e}")
        return {
            "themes": [],
            "thematic_results": {},
            "error": str(e)
        }


@tool
def synthesis_tool(themes: List[Dict], extracted_data: Dict, research_questions: List[str], 
                   user_prompt: str) -> Dict[str, Any]:
    """Synthesis Agent Tool"""
    print("---TOOL: Synthesis Agent---")
    try:
        synthesis_input = {
            "themes": themes,
            "extracted_data": extracted_data,
            "research_questions": research_questions,
            "user_prompt": user_prompt,
            "synthesis_style": "academic_literature_review"
        }
        print(f"Synthesizing {len(themes)} themes")
        synthesis_result = synthesis_agent.invoke(synthesis_input)
        print(f"Synthesis completed")
        return synthesis_result
    except Exception as e:
        print(f"Error in synthesis_tool: {e}")
        return {"synthesis_draft": f"Error: {e}", "synthesis_results": {}, "error": str(e)}

@tool
def quality_assurance_tool(synthesis_draft: str, extracted_data: Dict, 
                          research_questions: List[str]) -> Dict[str, Any]:
    """Quality Assurance Agent Tool"""
    print("---TOOL: Quality Assurance Agent---")
    try:
        qa_input = {
            "synthesis_draft": synthesis_draft,
            "extracted_data": extracted_data,
            "research_questions": research_questions,
            "quality_criteria": {
                "coherence_check": True,
                "citation_check": True,
                "plagiarism_check": True,
                "completeness_check": True
            }
        }
        print(f"Running quality checks")
        qa_result = qa_agent.invoke(qa_input)
        print(f"QA completed")
        return qa_result
    except Exception as e:
        print(f"Error in quality_assurance_tool: {e}")
        return {"quality_report": {"passed": False, "score": 0.0}, "quality_passed": False, "error": str(e)}

@tool
def generate_final_report(user_prompt: str, synthesis_draft: str, research_questions: List[str], 
                          quality_report: Dict, themes: List[Dict]) -> Dict[str, Any]:
    """Generates the final polished report"""
    print("---TOOL: Generating Final Report---")
    try:
        report_prompt = f"""Create a comprehensive literature review report based on:

            Original Query: {user_prompt}

            Research Questions:
            {chr(10).join(f"   {i+1}. {q}" for i, q in enumerate(research_questions))}

            Themes Identified: {len(themes)}
            Quality Score: {quality_report.get('score', 'N/A')}

            Synthesis:
            {synthesis_draft}

            Quality Assessment:
            {quality_report.get('summary', 'No assessment available')}

            Format with:
            1. Executive Summary
            2. Research Methodology
            3. Key Themes and Findings
            4. Quality Assessment
            5. Conclusions
            6. References
"""
        
        prompt = ChatPromptTemplate.from_template(report_prompt)
        chain = prompt | llm
        response = chain.invoke({})
        report = response.content
        
        print(f"Final report generated ({len(report)} characters)")
        return {"final_report": report, "workflow_complete": True}
        
    except Exception as e:
        print(f"Error in generate_final_report: {e}")
        return {"final_report": f"Error: {e}", "workflow_complete": True, "error": str(e)}

# Map tools
tools = [
    formulate_research_plan, search_and_filter_tool, screening_tool,
    extraction_tool, thematic_analysis_tool, synthesis_tool,
    quality_assurance_tool, generate_final_report
]
tool_map = {t.name: t for t in tools}

# --- 4. PLANNER & EXECUTOR NODES ---
PLANNER_PROMPT = """You are LitScout Orchestrator.
Review progress and decide which agent to invoke next.

**Current State**
User Query: {user_prompt}
Step: {current_step}
Research Questions: {research_questions_count}
Raw Papers: {raw_papers_count}
Filtered Papers: {filtered_papers_count}
Screened Papers: {screened_papers_count}
Themes: {themes_count}
Has Synthesis: {has_synthesis}
Quality Passed: {quality_passed}
Complete: {workflow_complete}

**Recent Activity**
{observations}

**Available Agents**
{tool_descriptions}

**Workflow**
1. formulate_research_plan
2. search_and_filter_tool
3. screening_tool
4. extraction_tool
5. thematic_analysis_tool
6. synthesis_tool
7. quality_assurance_tool
8. generate_final_report

Set next_agent to "END" when complete.
No emojis or text formatting.

{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)
planner_chain = prompt_template | llm | parser

def planner_node(state: LitScoutState) -> Dict[str, Any]:
    """Central planner node"""
    print("\n---ORCHESTRATOR: Planning---")
    try:
        response = planner_chain.invoke({
            "user_prompt": state.user_prompt,
            "current_step": state.current_step,
            "research_questions_count": len(state.research_questions),
            "raw_papers_count": len(state.raw_papers),
            "filtered_papers_count": len(state.filtered_papers),
            "screened_papers_count": len(state.screened_papers),
            "themes_count": len(state.themes),
            "has_synthesis": bool(state.synthesis_draft),
            "quality_passed": state.quality_passed,
            "workflow_complete": state.workflow_complete,
            "observations": "\n   ".join(state.observations[-3:]) if state.observations else "Starting workflow",
            "tool_descriptions": render_text_description(tools),
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"Plan: {' -> '.join(response.plan[:3])}{'...' if len(response.plan) > 3 else ''}")
        print(f"Next: {response.next_agent}")
        
        return {"plan": response.plan, "next_agent": response.next_agent}
    except Exception as e:
        print(f"Planner error: {e}")
        return {"plan": state.plan, "next_agent": "END", "error_messages": state.error_messages + [f"Planner error: {e}"]}

def tool_executor_node(state: LitScoutState) -> Dict[str, Any]:
    """Executes the selected agent"""
    tool_name = state.next_agent
    
    if tool_name not in tool_map:
        print(f"Unknown tool: {tool_name}")
        return {"error_messages": state.error_messages + [f"Unknown tool: {tool_name}"], "next_agent": "END"}
    
    selected_tool = tool_map[tool_name]
    print(f"\nExecuting: {tool_name}")
    
    try:
        # Build tool inputs based on tool name
        if tool_name == "formulate_research_plan":
            result = selected_tool.invoke({"user_prompt": state.user_prompt})
        elif tool_name == "search_and_filter_tool":
            result = selected_tool.invoke({
                "research_questions": state.research_questions,
                "inclusion_criteria": state.inclusion_criteria,
                "exclusion_criteria": state.exclusion_criteria,
                "user_prompt": state.user_prompt
            })
        elif tool_name == "screening_tool":
            result = selected_tool.invoke({
                "filtered_papers": state.filtered_papers,
                "research_questions": state.research_questions,
                "inclusion_criteria": state.inclusion_criteria,
                "exclusion_criteria": state.exclusion_criteria
            })
        elif tool_name == "extraction_tool":
            result = selected_tool.invoke({
                "screened_papers": state.screened_papers,
                "research_questions": state.research_questions
            })
        elif tool_name == "thematic_analysis_tool":
            result = selected_tool.invoke({
                "extracted_data": state.extracted_data,
                "research_questions": state.research_questions
            })
        elif tool_name == "synthesis_tool":
            result = selected_tool.invoke({
                "themes": state.themes,
                "extracted_data": state.extracted_data,
                "research_questions": state.research_questions,
                "user_prompt": state.user_prompt
            })
        elif tool_name == "quality_assurance_tool":
             result = selected_tool.invoke({
                "synthesis_draft": state.synthesis_draft,
                "extracted_data": state.extracted_data,
                "research_questions": state.research_questions
            })
        elif tool_name == "generate_final_report":
            result = selected_tool.invoke({
                "user_prompt": state.user_prompt,
                "synthesis_draft": state.synthesis_draft,
                "research_questions": state.research_questions,
                "quality_report": state.quality_report,
                "themes": state.themes
            })
        else:
            result = {"error": f"No execution logic for: {tool_name}"}
        
        observation = f"Executed {tool_name}"
        if "error" in result:
            observation += f" (with errors)"
        
        state_update = result.copy()
        state_update["observations"] = state.observations + [observation]
        state_update["current_step"] = state.current_step + 1
        
        return state_update
        
    except Exception as e:
        error_msg = f"Error in {tool_name}: {e}"
        print(f"Error: {error_msg}")
        return {
            "observations": state.observations + [error_msg],
            "error_messages": state.error_messages + [error_msg],
            "current_step": state.current_step + 1
        }
# --- 5. GRAPH ASSEMBLY ---
def router(state: LitScoutState) -> str:
    """Router for workflow"""
    if state.next_agent == "END" or state.workflow_complete:
        return "END"
    elif state.next_agent in tool_map:
        return "execute_tool"
    else:
        print(f"Unknown agent, ending")
        return "END"

workflow = StateGraph(LitScoutState)
workflow.add_node("planner", planner_node)
workflow.add_node("execute_tool", tool_executor_node)
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", router, {"execute_tool": "execute_tool", "END": END})
workflow.add_edge("execute_tool", "planner")
app = workflow.compile()


# --- 6. DATA FORMATTER (CRUCIAL FOR REACT UI) ---
def format_stage_1_planner(state: dict) -> dict:
    qs = state.get("research_questions") or []
    return {
        "id": "stage_1",
        "name": "Research Planner",
        "role": "Strategy",
        "status": "completed" if qs else "loading",
        "stats": f"{len(qs)} Questions",
        "data": {
            "questions": qs,
            "criteria": {"inclusion": state.get("inclusion_criteria", "N/A"), "exclusion": state.get("exclusion_criteria", "N/A")}
        }
    }

def format_stage_2_search(state: dict) -> dict:
    raw = state.get("raw_papers", [])
    filtered = state.get("filtered_papers", [])
    status = "completed" if filtered else "loading" if raw else "pending"
    return {
        "id": "stage_2",
        "name": "Deep Crawler",
        "role": "Search & Filter",
        "status": status,
        "stats": f"Found {len(raw)} | Kept {len(filtered)}",
        "data": {
            "summary": {"total": len(raw), "selected": len(filtered), "rejected": len(raw) - len(filtered)},
            "papers": filtered[:10] 
        }
    }
def format_stage_3_screening(state: dict):
    input_papers = state.get("filtered_papers", [])
    output_papers = state.get("screened_papers", [])
    
    input_count = len(input_papers)
    output_count = len(output_papers)
    exclusion_rate = ((input_count - output_count) / input_count * 100) if input_count > 0 else 0

    stats_string = f"In: {input_count} | Out: {output_count} | Excl: {exclusion_rate:.1f}%"

    return {
        "id": "stage_3",
        "name": "Semantic Screen",
        "role": "Screening",
        "status": "completed" if output_count > 0 else "loading",
        "stats": stats_string, 
        "data": {
            # --- THIS IS WHAT SHOWS UP IN THE LOWER PANEL ---
            "summary": {
                "total": input_count,
                "passed": output_count,
                "exclusion": f"{exclusion_rate:.1f}%",
                "method": "LLM-based Semantic Filtering"
            },
            "papers": output_papers[:15] 
        }
    }
async def run_orchestrator(user_prompt: str):
    # Initialize fresh state
    initial_state = LitScoutState(user_prompt=user_prompt)
    yield {"type": "log", "message": f"🚀 Initializing orchestrator..."}
    
    accumulated_state = {}
    
    try:
        # Increased recursion limit for deep research tasks
        async for step in app.astream(initial_state, {"recursion_limit": 50}):
            for node, output in step.items():
               # yield {"type": "log", "message": f"✅ Agent '{node}' finished step."} 
                
                if isinstance(output, dict):
                    accumulated_state.update(output)
                
                # --- LIVE UI UPDATE LOGIC ---
                current_steps = []
                
                # Stage 1: Planner
                if accumulated_state.get("research_questions"):
                    current_steps.append(format_stage_1_planner(accumulated_state))
                
                # Stage 2: Search
                if accumulated_state.get("raw_papers") or accumulated_state.get("filtered_papers"):
                    current_steps.append(format_stage_2_search(accumulated_state))
                
                # Stage 3: SCREENING (The new addition)
                if accumulated_state.get("screened_papers"):
                    current_steps.append(format_stage_3_screening(accumulated_state))

                if current_steps:
                    yield {
                        "type": "final",
                        "report": "Analyzing paper relevance and screening content...",
                        "steps": current_steps
                    }
            
        # Final Package after graph finishes
        if accumulated_state:
            yield {
                "type": "final",
                "report": accumulated_state.get("final_report", "Research complete."),
                "steps": [
                    format_stage_1_planner(accumulated_state),
                    format_stage_2_search(accumulated_state),
                    format_stage_3_screening(accumulated_state) # Final update
                ]
            }
    
    except Exception as e:
        yield {"type": "log", "message": f"⚠️ Notice: {str(e)}"}
        if accumulated_state:
            yield {
                "type": "final",
                "report": "Process paused. Results gathered so far are displayed.",
                "steps": [
                    format_stage_1_planner(accumulated_state), 
                    format_stage_2_search(accumulated_state),
                    format_stage_3_screening(accumulated_state)
                ]
            }