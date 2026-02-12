"""
Adaptive Paper Extraction Module
Implements LLM-driven adaptive paper limit with feedback loop.
"""

import json
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0
)

# Configuration
MIN_REQUIRED_PAPERS = int(os.getenv("MIN_REQUIRED_PAPERS", "40"))
MAX_ITERATIONS = int(os.getenv("MAX_EXTRACTION_ITERATIONS", "5"))


def determine_initial_paper_limit(
    screened_papers: List[Dict],
    research_questions: List[str],
    total_screened: int
) -> Dict:
    """
    LLM analyzes screened papers and suggests optimal initial limit.
    
    Args:
        screened_papers: Top papers with scores
        research_questions: Research questions for context
        total_screened: Total number of screened papers
    
    Returns:
        Dict with suggested_limit, reasoning, and confidence
    """
    # Sample top papers for analysis
    sample_papers = screened_papers[:10]
    sample_titles = [p.get('title', 'Unknown')[:80] for p in sample_papers]
    
    prompt = f"""You are optimizing a literature review pipeline.

Total screened papers: {total_screened}
Research questions:
{chr(10).join(f"- {q}" for q in research_questions)}

Top 10 paper titles:
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(sample_titles))}

CONTEXT:
- Typical PDF extraction success rate: 70-80%
- Target: At least {MIN_REQUIRED_PAPERS} papers with full text extracted
- We can iterate up to {MAX_ITERATIONS} times if needed

TASK:
Suggest an initial paper limit to attempt extraction.

CONSIDERATIONS:
1. Query specificity:
   - Narrow/specific topics (e.g., "federated learning for medical imaging"): 60-80 papers
   - Moderate topics: 80-120 papers
   - Broad topics (e.g., "machine learning"): 120-150 papers

2. Extraction challenges:
   - Medical papers: Often paywalled, suggest +20% buffer
   - ArXiv-heavy fields (CS, Physics): Higher success rate, can be conservative
   - Older papers (pre-2015): Lower availability, suggest +30% buffer

3. Safety margin:
   - Account for 70-80% success rate
   - Example: Need 40 papers → suggest 55-60 to account for failures

Respond with JSON only:
{{
  "suggested_limit": <int>,
  "reasoning": "<brief explanation>",
  "confidence": <float 0.0-1.0>
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        decision = json.loads(content)
        
        # Safety bounds
        suggested = decision["suggested_limit"]
        suggested = max(MIN_REQUIRED_PAPERS, min(suggested, total_screened))
        decision["suggested_limit"] = suggested
        
        return decision
    except Exception as e:
        print(f"Warning: LLM limit determination failed: {e}")
        # Fallback to conservative estimate
        return {
            "suggested_limit": min(100, total_screened),
            "reasoning": "Fallback: Conservative estimate due to LLM error",
            "confidence": 0.5
        }


def adjust_paper_limit(
    current_limit: int,
    extracted_count: int,
    failed_count: int,
    failure_reasons: Dict[str, int],
    iteration: int,
    total_available: int
) -> Dict:
    """
    LLM adjusts limit based on extraction results.
    
    Args:
        current_limit: Current paper limit attempted
        extracted_count: Successfully extracted papers
        failed_count: Failed extractions
        failure_reasons: Breakdown of failure types
        iteration: Current iteration number
        total_available: Total screened papers available
    
    Returns:
        Dict with new_limit, reasoning, and should_continue
    """
    success_rate = extracted_count / current_limit if current_limit > 0 else 0
    remaining_iterations = MAX_ITERATIONS - iteration
    papers_needed = max(0, MIN_REQUIRED_PAPERS - extracted_count)
    
    prompt = f"""You are adjusting the paper extraction limit based on results.

ITERATION {iteration} RESULTS:
- Attempted: {current_limit} papers
- Successfully extracted: {extracted_count} ({success_rate:.1%})
- Failed: {failed_count}

FAILURE BREAKDOWN:
- Paywall/403 errors: {failure_reasons.get('paywall', 0)}
- ArXiv not found: {failure_reasons.get('arxiv_not_found', 0)}
- Broken/invalid links: {failure_reasons.get('broken_link', 0)}
- Other errors: {failure_reasons.get('other', 0)}

TARGET & CONSTRAINTS:
- Target: At least {MIN_REQUIRED_PAPERS} papers with full text
- Current: {extracted_count} papers
- Still needed: {papers_needed} papers
- Remaining iterations: {remaining_iterations}
- Total papers available: {total_available}

DECISION LOGIC:
1. If we've reached target ({extracted_count} ≥ {MIN_REQUIRED_PAPERS}):
   → Set should_continue = false

2. If success rate is good (>70%) but need more papers:
   → Increase conservatively: +20-30%
   → new_limit = current_limit + (papers_needed / 0.7)

3. If mostly paywall failures (>50% of failures):
   → Increase aggressively: +50-70%
   → Reason: Need to skip paywalled papers

4. If mostly ArXiv/broken link failures:
   → Increase moderately: +30-40%
   → Reason: Some papers genuinely unavailable

5. If we've exhausted available papers:
   → Set should_continue = false

6. If remaining iterations is low:
   → Be more aggressive to reach target quickly

Respond with JSON only:
{{
  "new_limit": <int>,
  "reasoning": "<brief explanation>",
  "should_continue": <bool>
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        decision = json.loads(content)
        
        # Safety bounds
        new_limit = decision["new_limit"]
        new_limit = max(current_limit, min(new_limit, total_available))
        decision["new_limit"] = new_limit
        
        # Force stop if exhausted or reached target
        if new_limit >= total_available:
            print(f"\n⚠️ Reached maximum available papers ({total_available})")
            decision["should_continue"] = False
            decision["reasoning"] += " | Forced stop: exhausted all papers"
        
        if extracted_count >= MIN_REQUIRED_PAPERS:
            print(f"\n✅ Target already met ({extracted_count} ≥ {MIN_REQUIRED_PAPERS})")
            decision["should_continue"] = False
            decision["reasoning"] += " | Forced stop: target met"
        
        # Log LLM decision
        print(f"\n🤖 LLM DECISION:")
        print(f"   New limit: {decision['new_limit']}")
        print(f"   Should continue: {decision['should_continue']}")
        print(f"   Reasoning: {decision['reasoning']}")
        
        return decision
    except Exception as e:
        print(f"Warning: LLM adjustment failed: {e}")
        # Fallback: increase by 30%
        new_limit = min(int(current_limit * 1.3), total_available)
        should_continue = extracted_count < MIN_REQUIRED_PAPERS and new_limit < total_available
        
        print(f"\n⚠️ FALLBACK DECISION (LLM failed):")
        print(f"   New limit: {new_limit}")
        print(f"   Should continue: {should_continue}")
        
        return {
            "new_limit": new_limit,
            "reasoning": f"Fallback: Increase by 30% due to LLM error: {str(e)}",
            "should_continue": should_continue
        }


def categorize_extraction_failures(extraction_results: Dict) -> Dict[str, int]:
    """
    Categorize extraction failures by type.
    
    Args:
        extraction_results: Results from extraction agent
    
    Returns:
        Dict with counts of each failure type
    """
    failures = {
        'paywall': 0,
        'arxiv_not_found': 0,
        'broken_link': 0,
        'other': 0
    }
    
    for paper_id, result in extraction_results.items():
        status = result.get('status', 'unknown')
        message = result.get('message', '').lower()
        
        if status == 'failed':
            if '403' in message or 'forbidden' in message or 'paywall' in message:
                failures['paywall'] += 1
            elif 'arxiv' in message and ('not found' in message or 'no match' in message):
                failures['arxiv_not_found'] += 1
            elif '404' in message or 'broken' in message or 'invalid' in message:
                failures['broken_link'] += 1
            else:
                failures['other'] += 1
    
    return failures


def adaptive_extraction_loop(
    screened_papers: List[Dict],
    research_questions: List[str],
    extraction_agent
) -> Tuple[List[Dict], Dict]:
    """
    Iteratively extract papers with LLM-guided limit adjustment.
    
    Args:
        screened_papers: All screened papers (sorted by relevance)
        research_questions: Research questions for context
        extraction_agent: The extraction agent to use
    
    Returns:
        Tuple of (extracted_papers, metadata)
    """
    print("\n" + "="*80)
    print("🔄 ADAPTIVE EXTRACTION LOOP")
    print("="*80)
    print(f"Total screened papers: {len(screened_papers)}")
    print(f"Target: At least {MIN_REQUIRED_PAPERS} papers with full text")
    print(f"Max iterations: {MAX_ITERATIONS}")
    
    iteration = 0
    all_extracted_papers = []
    extraction_metadata = {
        'iterations': [],
        'total_attempted': 0,
        'total_extracted': 0,
        'final_limit': 0
    }
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"📊 ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*80}")
        
        # Determine limit for this iteration
        if iteration == 1:
            # Initial limit from LLM
            decision = determine_initial_paper_limit(
                screened_papers,
                research_questions,
                len(screened_papers)
            )
            limit = decision["suggested_limit"]
            print(f"\n🤖 LLM Initial Suggestion:")
            print(f"   Limit: {limit} papers")
            print(f"   Reasoning: {decision['reasoning']}")
            print(f"   Confidence: {decision['confidence']:.1%}")
        else:
            # Adjust based on previous results
            prev_iter = extraction_metadata['iterations'][-1]
            decision = adjust_paper_limit(
                current_limit=prev_iter['limit'],
                extracted_count=len(all_extracted_papers),  # Cumulative total
                failed_count=prev_iter['failed_this_iter'],  # Failures from last iteration
                failure_reasons=prev_iter['failure_breakdown'],
                iteration=iteration,
                total_available=len(screened_papers)
            )
            limit = decision["new_limit"]
            print(f"\n🤖 LLM Adjustment:")
            print(f"   New limit: {limit} papers (was {prev_iter['limit']})")
            print(f"   Reasoning: {decision['reasoning']}")
            
            if not decision.get('should_continue', True):
                print(f"\n✅ LLM recommends stopping")
                break
        
        # Extract PDFs for NEW papers only (incremental extraction)
        if iteration == 1:
            # First iteration: extract from beginning
            papers_to_extract = screened_papers[:limit]
            previous_limit = 0
            print(f"\n📥 Extracting papers 1-{limit}...")
        else:
            # Subsequent iterations: only extract NEW papers
            prev_iter = extraction_metadata['iterations'][-1]
            previous_limit = prev_iter['limit']
            
            if limit > previous_limit:
                # Extract only the new papers (from previous_limit+1 to limit)
                papers_to_extract = screened_papers[previous_limit:limit]
                print(f"\n📥 Extracting NEW papers {previous_limit+1}-{limit} ({len(papers_to_extract)} papers)...")
            else:
                # No new papers to extract (limit didn't increase)
                print(f"\n⚠️ No new papers to extract (limit {limit} <= previous {previous_limit})")
                break
        
        try:
            extraction_input = {
                "screened_papers": papers_to_extract,
                "papers_with_pdfs": [],
                "papers_with_text": [],
                "extraction_results": {}
            }
            extraction_result = extraction_agent.invoke(extraction_input)
            
            # Get results
            newly_extracted_papers = extraction_result.get('papers_with_text', [])
            extraction_results = extraction_result.get('extraction_results', {})
            
            # Accumulate extracted papers (add new ones to existing)
            if iteration == 1:
                all_extracted_papers = newly_extracted_papers
            else:
                all_extracted_papers.extend(newly_extracted_papers)
            
            # Categorize failures (only for this batch)
            failure_breakdown = categorize_extraction_failures(extraction_results)
            
            # Update metadata
            iter_metadata = {
                'iteration': iteration,
                'limit': limit,
                'papers_attempted_this_iter': len(papers_to_extract),
                'extracted_this_iter': len(newly_extracted_papers),
                'failed_this_iter': len(papers_to_extract) - len(newly_extracted_papers),
                'cumulative_extracted': len(all_extracted_papers),
                'success_rate_this_iter': len(newly_extracted_papers) / len(papers_to_extract) if len(papers_to_extract) > 0 else 0,
                'failure_breakdown': failure_breakdown
            }
            extraction_metadata['iterations'].append(iter_metadata)
            extraction_metadata['total_attempted'] = limit
            extraction_metadata['final_limit'] = limit
            extraction_metadata['total_extracted'] = len(all_extracted_papers)
            
            # Print results
            print(f"\n📊 Results (this iteration):")
            print(f"   ✅ Extracted: {len(newly_extracted_papers)}/{len(papers_to_extract)} ({iter_metadata['success_rate_this_iter']:.1%})")
            print(f"   ❌ Failed: {iter_metadata['failed_this_iter']}")
            print(f"   📚 Cumulative total: {len(all_extracted_papers)} papers")
            print(f"   Failure breakdown:")
            for reason, count in failure_breakdown.items():
                if count > 0:
                    print(f"      - {reason}: {count}")
            
            # Check if we've reached target
            if len(all_extracted_papers) >= MIN_REQUIRED_PAPERS:
                print(f"\n✅ TARGET REACHED! ({len(all_extracted_papers)} ≥ {MIN_REQUIRED_PAPERS})")
                break
            
            # Check if we've exhausted all papers
            if limit >= len(screened_papers):
                print(f"\n⚠️ Reached maximum available papers ({len(screened_papers)})")
                break
                
        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final summary
    print(f"\n{'='*80}")
    print("📋 ADAPTIVE EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total iterations: {iteration}")
    print(f"Papers attempted: {extraction_metadata['total_attempted']}")
    print(f"Papers extracted: {extraction_metadata['total_extracted']}")
    print(f"Overall success rate: {extraction_metadata['total_extracted']/extraction_metadata['total_attempted']:.1%}")
    print(f"Target met: {'✅ Yes' if extraction_metadata['total_extracted'] >= MIN_REQUIRED_PAPERS else '❌ No'}")
    
    return all_extracted_papers, extraction_metadata
