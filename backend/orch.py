import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool, render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# --- MOCK AGENTS FOR DEVELOPMENT ---
class MockAgent:
    def invoke(self, input_data):
        return {"status": "mock_response", "data": input_data}


screening_agent = MockAgent()
extraction_agent = MockAgent()
thematic_agent = MockAgent()
synthesis_agent = MockAgent()
qa_agent = MockAgent()


# Import real agents
try:
    from agents.search_and_filter_agent import saf_agent
    from agents.screening_agent import screening_agent
except ImportError as e:
    print(f"Warning: Could not import real agents: {e}")


load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file.")


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


# --- 2. LLM AND PARSER SETUP ---

llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")

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
    Formulates a research plan based on the user_prompt. This is the first step.
    Generates research questions and defines inclusion/exclusion criteria using an LLM.
    """
    print("---TOOL: Formulating Research Plan---")
    
    try:
        plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        prompt_template_string = """Generate a research plan for a literature review.

            Research Prompt: {user_prompt}
            Generate:

            1. **research_questions**: 3-5 specific research questions

            2. **inclusion_criteria**: CONTENT-FOCUSED keywords that papers should contain.
            - Focus on: methods, techniques, domains, research areas
            

            3. **exclusion_criteria**: CONTENT to AVOID (methods/topics that are out of scope).
            - Focus on: alternative methods, unrelated topics, specific languages to exclude
            
            - DO NOT include metadata criteria
            - Should be DIFFERENT from inclusion criteria (not just negation)

            {format_instructions}
            """
        
        prompt = ChatPromptTemplate.from_template(
            prompt_template_string,
            partial_variables={"format_instructions": plan_parser.get_format_instructions()},
        )
        
        chain = prompt | llm | plan_parser
        response_obj = chain.invoke({"user_prompt": user_prompt})
        
        result = response_obj.model_dump()
        
        # Clean output formatting
        print(f"\nGenerated {len(result['research_questions'])} research questions:")
        for i, q in enumerate(result['research_questions'], 1):
            print(f"   {i}. {q}")
        
        print(f"\nInclusion criteria: {result['inclusion_criteria']}")
        print(f"Exclusion criteria: {result['exclusion_criteria']}\n")
        
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
    Extracts metadata filters (year range, sources) from user prompt.
    The SAF agent handles searching and metadata filtering.
    """
    print("---TOOL: Search and Filter Agent---")
    
    try:
        import datetime
        import re
        
        current_year = datetime.datetime.now().year
        
        # Extract year range and sources from user prompt using LLM
        extraction_prompt = f"""Extract filtering criteria from this research query:

                "{user_prompt}"

                Respond with ONLY a JSON object:
                {{
                "start_year": <number or null>,
                "end_year": <number or null>,
                "sources": [<list of publication venues/conferences or empty array>]
                }}

                Examples:
                - "papers from 2020 to 2023" → {{"start_year": 2020, "end_year": 2023, "sources": []}}
                - "recent studies" → {{"start_year": null, "end_year": null, "sources": []}}
                - "CVPR papers from last 3 years" → {{"start_year": {current_year - 3}, "end_year": {current_year}, "sources": ["CVPR"]}}

                If no specific years mentioned, use null for both start_year and end_year (default to last 5 years).
                """
        
        try:
            response = llm.invoke(extraction_prompt)
            content = response.content.strip().replace("```json", "").replace("```", "").strip()
            metadata = json.loads(content)
            
            start_year = metadata.get("start_year") or (current_year - 5)
            end_year = metadata.get("end_year") or current_year
            sources = metadata.get("sources", [])
            
        except Exception as e:
            print(f"Could not extract metadata, using defaults: {e}")
            start_year = current_year - 5
            end_year = current_year
            sources = []
        
        saf_input = {
            "research_questions": research_questions,
            "start_year": start_year,
            "end_year": end_year,
            "sources": sources,
            "raw_papers": [],
            "filtered_papers": []
        }
        
        print(f"Year range: {start_year}-{end_year}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        else:
            print(f"Sources: All venues")
        print(f"Processing {len(research_questions)} research questions")
        
        saf_result = saf_agent.invoke(saf_input)
        
        print(f"SAF agent completed: {len(saf_result.get('filtered_papers', []))} papers found")
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
    Screening Agent Tool - Delegates to the specialized screening agent.
    Performs content-based relevance screening of filtered papers.
    Metadata filtering is already done by SAF agent.
    """
    print("---TOOL: Screening Agent---")
    
    try:
        screening_input = {
            "filtered_papers": filtered_papers,
            "research_questions": research_questions,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            
            # Adaptive thresholds (will be set by LLM in screening agent)
            "keyword_high_threshold": 0.0,
            "keyword_medium_threshold": 0.0,
            "tfidf_threshold": 0.0,
            "use_llm_screening": False,
            
            # No metadata filters - SAF handles this
            "year_range": (1900, 2100),
            "allowed_publication_types": [],
            "min_citation_count": 0,
            
            # Internal state
            "metadata_filtered_papers": [],
            "high_relevance_papers": [],
            "medium_relevance_papers": [],
            "borderline_papers": [],
            "screened_papers": [],
            "screening_results": {}
        }
        
        print(f"Screening {len(filtered_papers)} papers")
        screening_result = screening_agent.invoke(screening_input)
        
        print(f"Screening completed: {len(screening_result.get('screened_papers', []))} papers passed")
        return screening_result
        
    except Exception as e:
        print(f"Error in screening_tool: {e}")
        return {
            "screened_papers": filtered_papers[:100] if filtered_papers else [],
            "screening_results": {"error": str(e)},
            "error": str(e)
        }

@tool
def extraction_tool(screened_papers: List[Dict], research_questions: List[str]) -> Dict[str, Any]:
    """
    Extraction Agent Tool - Delegates to the specialized extraction agent.
    Extracts structured data from screened papers using CORE API and LLMs.
    """
    print("---TOOL: Extraction Agent---")
    
    try:
        extraction_input = {
            "screened_papers": screened_papers,
            "research_questions": research_questions,
            "extraction_template": "standard_literature_review"
        }
        
        print(f"Invoking extraction agent for {len(screened_papers)} papers")
        extraction_result = extraction_agent.invoke(extraction_input)
        
        print(f"Extraction agent completed successfully")
        return extraction_result
        
    except Exception as e:
        print(f"Error in extraction_tool: {e}")
        return {
            "extracted_data": {},
            "extraction_results": {},
            "error": str(e)
        }

@tool
def thematic_analysis_tool(extracted_data: Dict, research_questions: List[str]) -> Dict[str, Any]:
    """
    Thematic Analysis Agent Tool - Delegates to the specialized thematic analysis agent.
    Identifies themes and patterns across the extracted data.
    """
    print("---TOOL: Thematic Analysis Agent---")
    
    try:
        thematic_input = {
            "extracted_data": extracted_data,
            "research_questions": research_questions,
            "clustering_method": "hierarchical",
            "min_theme_size": 3
        }
        
        print(f"🔍 Invoking thematic analysis agent")
        thematic_result = thematic_agent.invoke(thematic_input)
        
        print(f"Thematic analysis agent completed successfully")
        return thematic_result
        
    except Exception as e:
        print(f"Error in thematic_analysis_tool: {e}")
        return {
            "themes": [],
            "thematic_results": {},
            "error": str(e)
        }

@tool
def synthesis_tool(themes: List[Dict], extracted_data: Dict, research_questions: List[str], user_prompt: str) -> Dict[str, Any]:
    """
    Synthesis Agent Tool - Delegates to the specialized synthesis agent.
    Creates coherent literature review narrative from themes and extracted data.
    """
    print("---TOOL: Synthesis Agent---")
    
    try:
        synthesis_input = {
            "themes": themes,
            "extracted_data": extracted_data,
            "research_questions": research_questions,
            "user_prompt": user_prompt,
            "synthesis_style": "academic_literature_review"
        }
        
        print(f"Invoking synthesis agent with {len(themes)} themes")
        synthesis_result = synthesis_agent.invoke(synthesis_input)
        
        print(f"Synthesis agent completed successfully")
        return synthesis_result
        
    except Exception as e:
        print(f"Error in synthesis_tool: {e}")
        return {
            "synthesis_draft": f"Error in synthesis: {e}",
            "synthesis_results": {},
            "error": str(e)
        }

@tool
def quality_assurance_tool(synthesis_draft: str, extracted_data: Dict, research_questions: List[str]) -> Dict[str, Any]:
    """
    Quality Assurance Agent Tool - Delegates to the specialized QA agent.
    Performs comprehensive quality checks on the synthesized literature review.
    """
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
        
        print(f"Invoking quality assurance agent")
        qa_result = qa_agent.invoke(qa_input)
        
        print(f"Quality assurance agent completed successfully")
        return qa_result
        
    except Exception as e:
        print(f"❌ Error in quality_assurance_tool: {e}")
        return {
            "quality_report": {"passed": False, "score": 0.0},
            "quality_passed": False,
            "error": str(e)
        }

@tool
def generate_final_report(user_prompt: str, synthesis_draft: str, research_questions: List[str], 
                          quality_report: Dict, themes: List[Dict]) -> Dict[str, Any]:
    """
    Generates the final, polished report incorporating all analysis results.
    This is the orchestrator's final step that combines all agent outputs.
    """
    print("---TOOL: Generating Final Report---")
    
    try:
        report_prompt = f"""
        Create a comprehensive literature review report based on:
        
        Original Query: {user_prompt}
        
        Research Questions:
        {chr(10).join(f"   {i+1}. {q}" for i, q in enumerate(research_questions))}
        
        Number of Themes Identified: {len(themes)}
        Quality Score: {quality_report.get('score', 'N/A')}
        
        Synthesis Content:
        {synthesis_draft}
        
        Quality Assessment:
        {quality_report.get('summary', 'No quality assessment available')}
        
        Format the report with:
        1. Executive Summary
        2. Research Methodology
        3. Key Themes and Findings
        4. Quality Assessment
        5. Conclusions and Future Research Directions
        6. References and Acknowledgments
        """
        
        prompt = ChatPromptTemplate.from_template(report_prompt)
        chain = prompt | llm
        
        response = chain.invoke({})
        report = response.content
        
        print(f"Generated final report ({len(report)} characters)")
        
        return {
            "final_report": report,
            "workflow_complete": True
        }
        
    except Exception as e:
        print(f"Error in generate_final_report: {e}")
        return {
            "final_report": f"Error generating final report: {e}",
            "workflow_complete": True,
            "error": str(e)
        }


# Map tools for easy access
tools = [
    formulate_research_plan, 
    search_and_filter_tool, 
    screening_tool,
    extraction_tool,
    thematic_analysis_tool,
    synthesis_tool,
    quality_assurance_tool,
    generate_final_report
]
tool_map = {t.name: t for t in tools}


# --- 4. PLANNER & EXECUTOR NODES ---

PLANNER_PROMPT = """You are LitScout Orchestrator, an autonomous AI research assistant coordinator.
Your goal is to orchestrate a complete literature review using specialized agents.
Review your current progress and decide which specialized agent to invoke next.

**Current State & Progress**
User Query: {user_prompt}
Current Step: {current_step}
Research Questions: {research_questions_count} formulated
Raw Papers Found: {raw_papers_count}
Filtered Papers: {filtered_papers_count}
Screened Papers: {screened_papers_count}
Themes Identified: {themes_count}
Has Synthesis: {has_synthesis}
Quality Passed: {quality_passed}
Workflow Complete: {workflow_complete}

**Latest Observations**
{observations}

**Available Specialized Agents (Tools)**
{tool_descriptions}

**Instructions**
Follow the logical workflow sequence:
  1. formulate_research_plan
  2. search_and_filter_tool (now handles metadata filtering internally)
  3. screening_tool (content-based screening only)
  4. extraction_tool
  5. thematic_analysis_tool
  6. synthesis_tool
  7. quality_assurance_tool
  8. generate_final_report

- If quality assurance fails, you may need to go back to synthesis
- If screening produces too few papers, you may need to go back to search_and_filter
- Only proceed to the next step if prerequisites are met
- Set next_agent to "END" when workflow is complete
- The plan should contain what you have done till now and how much is to complete.
- Never add any emojies. Dont apply any text formatting like bold,italics.

{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)
planner_chain = prompt_template | llm | parser

def planner_node(state: LitScoutState) -> Dict[str, Any]:
    """The central planner node that coordinates all specialized agents."""
    print("\n---ORCHESTRATOR: Planning next action---")
    
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
            "observations": "\n   ".join(state.observations[-3:]) if state.observations else "No observations yet.",
            "tool_descriptions": render_text_description(tools),
            "format_instructions": parser.get_format_instructions()
        })
        print("Plan:")
        for step in response.plan:
            print(step)
        print(f"   Next action: {response.next_agent}")
        
        return {
            "plan": response.plan,
            "next_agent": response.next_agent
        }
        
    except Exception as e:
        print(f"Planner error: {e}")
        return {
            "plan": state.plan,
            "next_agent": "END",
            "error_messages": state.error_messages + [f"Planner error: {e}"]
        }

def tool_executor_node(state: LitScoutState) -> Dict[str, Any]:
    """Executes the specialized agent selected by the planner."""
    tool_name = state.next_agent
    
    if tool_name not in tool_map:
        error_msg = f"Unknown agent/tool: {tool_name}"
        print(f"{error_msg}")
        return {
            "error_messages": state.error_messages + [error_msg],
            "next_agent": "END"
        }
    
    selected_tool = tool_map[tool_name]
    print(f"---ORCHESTRATOR EXECUTOR: Invoking '{tool_name}'---")
    
    try:
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
            result = {"error": f"No execution logic for agent: {tool_name}"}
        
        observation = f"Successfully invoked agent '{tool_name}'"
        if "error" in result:
            observation += f" with errors: {result['error']}"
        
        state_update = result.copy()
        state_update["observations"] = state.observations + [observation]
        state_update["current_step"] = state.current_step + 1
        
        print(f"{observation}")
        return state_update
        
    except Exception as e:
        error_msg = f"Error invoking agent '{tool_name}': {e}"
        print(f"{error_msg}")
        return {
            "observations": state.observations + [error_msg],
            "error_messages": state.error_messages + [error_msg],
            "current_step": state.current_step + 1
        }


# --- 5. GRAPH ASSEMBLY ---

def router(state: LitScoutState) -> str:
    """The router that decides the next step in the orchestration."""
    if state.next_agent == "END" or state.workflow_complete:
        print("---ORCHESTRATOR ROUTER: Workflow complete. Ending process.---")
        return "END"
    elif state.next_agent in tool_map:
        return "execute_tool"
    else:
        print(f"---ORCHESTRATOR ROUTER: Unknown agent '{state.next_agent}', ending process.---")
        return "END"

# Create the orchestrator workflow
workflow = StateGraph(LitScoutState)
workflow.add_node("planner", planner_node)
workflow.add_node("execute_tool", tool_executor_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", 
    router, 
    {
        "execute_tool": "execute_tool", 
        "END": END
    }
)
workflow.add_edge("execute_tool", "planner")

app = workflow.compile()


# --- 6. DATA FORMATTER (CRUCIAL FOR REACT UI) ---

def format_stage_1_planner(state: dict) -> dict:
    """
    Extracts REAL questions and criteria generated by the LLM from the state 
    and formats them into the exact JSON structure the React Dashboard expects.
    """
    questions = state.get("research_questions", [])
    inc_criteria = state.get("inclusion_criteria", "N/A")
    exc_criteria = state.get("exclusion_criteria", "N/A")

    return {
        "id": "stage_1",
        "name": "Research Planner",
        "role": "Strategy",  # Matches the IF condition in Dashboard.js
        "status": "completed",
        "stats": f"{len(questions)} Questions Formulated",
        "data": {
            "questions": questions,
            "criteria": {
                "inclusion": inc_criteria,
                "exclusion": exc_criteria
            }
        }
    }


# --- 7. EXPORTED EXECUTION FUNCTION ---

async def run_orchestrator(user_prompt: str):
    """
    Function to be called by the FastAPI backend.
    """
    initial_state = LitScoutState(
        user_prompt=user_prompt
    )
    
    print(f"--- STARTING ORCHESTRATION FOR: {user_prompt} ---")
    
    try:
        final_state = None
        
        # Run the Graph
        # We iterate to ensure the graph executes fully and we capture the final state
        async for step in app.astream(initial_state, {"recursion_limit": 25}):
            node, output = next(iter(step.items()))
            print(f"Executed: {node}")
            final_state = output
            
        if final_state:
            # --- 1. FORMAT STAGE 1 WITH REAL DATA ---
            # This gets the actual questions/criteria from the LLM
            stage_1_real = format_stage_1_planner(final_state)
            
            # --- 2. ADD PLACEHOLDERS FOR OTHER STAGES ---
            # We add these empty/pending stages so the React Dashboard 
            # renders the full 8-button workflow bar correctly.
            placeholders = [
                {"id": "stage_2", "name": "Deep Crawler", "role": "Search & Filter", "status": "pending", "stats": "Waiting...", "data": {"summary": {"total": 0, "selected": 0, "rejected": 0}, "papers": []}},
                {"id": "stage_3", "name": "Semantic Screen", "role": "Screening", "status": "pending", "stats": "Waiting...", "data": {"papers": []}},
                {"id": "stage_4", "name": "Data Extractor", "role": "Extraction", "status": "pending", "stats": "Waiting...", "data": {}},
                {"id": "stage_5", "name": "Theme Mapper", "role": "Thematic Analysis", "status": "pending", "stats": "Waiting...", "data": {"themes": []}},
                {"id": "stage_6", "name": "Synthesizer", "role": "Drafting", "status": "pending", "stats": "Waiting...", "data": {}},
                {"id": "stage_7", "name": "QA Bot", "role": "Quality Assurance", "status": "pending", "stats": "Waiting...", "data": {"checklist": []}},
                {"id": "stage_8", "name": "Publisher", "role": "Final Report", "status": "pending", "stats": "Waiting...", "data": {}}
            ]
            
            # Combine Stage 1 (Real) + Stages 2-8 (Placeholders)
            workflow_steps = [stage_1_real] + placeholders

            return {
                "status": "success", 
                "report": final_state.get("final_report", "No report generated."),
                "steps": workflow_steps # <--- Sends the list to React
            }
        else:
            return {
                "status": "error", 
                "report": "Workflow finished but no final report was generated.",
                "details": "Unknown error"
            }

    except Exception as e:
        print(f"Orchestration Error: {e}")
        return {"status": "error", "report": str(e)}