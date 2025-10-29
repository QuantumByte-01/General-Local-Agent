from typing import List, Dict, Tuple
from .state import AgentState, ls_start_run, ls_end_run
from .graph import build_agent_graph


def run_agent_once(
    goal: str,
    chat_history: List[Dict[str, str]],
    base_dir: str,
    max_retries: int = 3,
) -> Tuple[str, str]:
  

    parent_span = ls_start_run(
        name="LangGraph",
        run_type="chain",
        inputs={
            "goal": goal,
            "base_dir": base_dir,
            "chat_history_tail": chat_history[-4:],
        },
        parent_run_id=None,
    )

    graph = build_agent_graph()

    init_state: AgentState = {
        "goal": goal,
        "base_dir": base_dir,
        "chat_history": chat_history,
        "max_retries": max_retries,
        "parent_run_id": parent_span.get("run_id"),
    }

    final_state: AgentState = graph.invoke(init_state)

    final_answer = final_state.get("final_answer", "").strip()
    results_for_final = final_state.get("results_for_final", [])
    global_success = final_state.get("global_success_criteria", "")
    subtasks_list = final_state.get("subtasks", [])

    ls_end_run(
        parent_span,
        outputs={
            "final_answer": final_answer,
            "global_success_criteria": global_success,
            "subtask_count": len(subtasks_list),
        },
        error=None,
    )

    dbg_lines = []
    for r in results_for_final:
        dbg_lines.append(
            f"[Subtask {r.get('id')} success={r.get('success')} "
            f"reason='{r.get('reason','')[:80]}']"
        )
    debug_info = "\n".join(dbg_lines)

    return final_answer, debug_info
