import os
from datetime import datetime
from typing import TypedDict, List, Dict, Any


try:
    from langsmith import Client as LangsmithClient
except Exception:
    LangsmithClient = None


def ls_start_run(name: str, run_type: str, inputs: Dict[str, Any], parent_run_id=None):
  
    handle = {"client": None, "run_id": None}
    if LangsmithClient is None:
        return handle
    try:
        client = LangsmithClient()
        run = client.create_run(
            name=name,
            run_type=run_type,
            inputs=inputs,
            parent_run_id=parent_run_id,
            project_name=os.getenv("LANGSMITH_PROJECT")
                or os.getenv("LANGCHAIN_PROJECT"),
            start_time=datetime.utcnow(),
        )
        handle = {"client": client, "run_id": run.id}
    except Exception:
        pass
    return handle


def ls_end_run(handle: Dict[str, Any], outputs: Dict[str, Any], error: str = None):
    
    client = handle.get("client")
    run_id = handle.get("run_id")
    if not client or not run_id:
        return
    try:
        client.update_run(
            run_id=run_id,
            outputs=outputs,
            error=error,
            end_time=datetime.utcnow(),
        )
    except Exception:
        pass



class AgentState(TypedDict, total=False):
    # context
    goal: str
    base_dir: str
    chat_history: List[Dict[str, str]]

    # Plan info
    plan: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    global_success_criteria: str

    # Progress
    cursor: int          
    attempt: int        
    max_retries: int

    # Last execution attempt info
    last_stdout: str
    last_stderr: str
    last_reason: str
    last_success: bool

    # Aggregate results for final summary
    results_for_final: List[Dict[str, Any]]
    final_answer: str

    # Routing hint for conditional edges
    route: str

    # LangSmith parent
    parent_run_id: str

    # Flags
    preplanned: bool
