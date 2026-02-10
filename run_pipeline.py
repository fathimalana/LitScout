
import asyncio
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.orch import run_orchestrator

async def main():
    topic = "Automating SLRs using Agentic AI and LLMs"
    print(f"Starting LitScout Pipeline for topic: {topic}")
    print("="*80)
    
    async for event in run_orchestrator(topic):
        if event["type"] == "log":
            print(f"[LOG] {event['message']}")
        elif event["type"] == "final":
            print(f"\n[UPDATE] {event['report']}")
            # Start printing step details if available to see progress
            if "steps" in event:
                last_step = event["steps"][-1]
                print(f"   Current Stage: {last_step['name']} ({last_step['status']})")
                if "data" in last_step: # Print summary stats if available
                     if "stats" in last_step:
                         print(f"   Stats: {last_step['stats']}")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
