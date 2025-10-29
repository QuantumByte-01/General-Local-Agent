import os
import subprocess
import tempfile
from typing import Dict, Any, List

from .state import AgentState, ls_start_run, ls_end_run
from .llm import (
    plan_tasks,
    assess_success,
    repair_command,
    summarize_final,
)




def _run_cmd_once(command: str) -> Dict[str, Any]:
    """
    Run ONE Windows shell / PowerShell command.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Exception while running command: {e}",
        }


def _run_python_once(code_str: str) -> Dict[str, Any]:
    """
    Write code_str to a temp file, run python temp.py, capture output.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
            f.write(code_str)

        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=20,
        )

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Exception while running python code: {e}",
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _exec_once(language: str, command: str) -> Dict[str, Any]:
    """
    Dispatch to cmd or python executor.
    """
    if language == "python":
        return _run_python_once(command)
    else:
        return _run_cmd_once(command)




def plan_node(state: AgentState) -> AgentState:
    """
    Call Gemini planner to produce subtasks + success criteria.
    Initialize cursor/attempt/etc.
    Also open a 'plan_tasks' child span in LangSmith.
    """
    parent_id = state.get("parent_run_id")

    plan_span = ls_start_run(
        name="plan_tasks",
        run_type="llm",
        inputs={
            "goal": state["goal"],
            "base_dir": state["base_dir"],
        },
        parent_run_id=parent_id,
    )

    plan = plan_tasks(
        chat_history=state["chat_history"],
        goal=state["goal"],
        base_dir=state["base_dir"],
    )

    ls_end_run(plan_span, outputs={"plan": plan}, error=None)

    # init loop state
    return {
        "plan": plan,
        "subtasks": plan.get("subtasks", []),
        "global_success_criteria": plan.get("global_success_criteria", ""),
        "cursor": 0,
        "attempt": 0,
        "results_for_final": [],
        "last_stdout": "",
        "last_stderr": "",
        "last_reason": "",
        "last_success": False,
        "route": "",
    }


def execute_node(state: AgentState) -> AgentState:
    """
    Execute the current subtask command/script.
    Increment attempt.
    Trace child span 'execute'.
    """
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0) + 1  

    if cursor < 0 or cursor >= len(subtasks):
        # nothing to run
        return {
            "attempt": attempt,
            "last_stdout": "",
            "last_stderr": "",
        }

    st = subtasks[cursor]
    lang = st.get("language", "cmd")
    cmd = st.get("command", "")
    desc = st.get("description", "")

    exec_span = ls_start_run(
        name="execute",
        run_type="tool",
        inputs={
            "subtask_id": st.get("id"),
            "attempt": attempt,
            "description": desc,
            "language": lang,
            "command": cmd,
        },
        parent_run_id=state.get("parent_run_id"),
    )

    res = _exec_once(lang, cmd)
    stdout = res.get("stdout", "")
    stderr = res.get("stderr", "")
    rc = res.get("returncode", 0)

    ls_end_run(
        exec_span,
        outputs={
            "returncode": rc,
            "stdout": stdout[:1000],
            "stderr": stderr[:500],
        },
        error=None,
    )

    return {
        "attempt": attempt,
        "last_stdout": stdout,
        "last_stderr": stderr,
    }


def verify_node(state: AgentState) -> AgentState:
    """
    Ask Gemini if success criteria was met.
    If yes, append a success entry into results_for_final.
    Trace child span 'verify'.
    """
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0)
    results = state.get("results_for_final", [])

    if cursor < 0 or cursor >= len(subtasks):
        return {
            "last_success": True,
            "last_reason": "No subtask at cursor.",
            "results_for_final": results,
        }

    st = subtasks[cursor]
    sid = st.get("id")
    desc = st.get("description", "")
    crit = st.get("success_criteria", "")

    verify_span = ls_start_run(
        name="verify",
        run_type="llm",
        inputs={
            "subtask_id": sid,
            "attempt": attempt,
            "description": desc,
            "success_criteria": crit,
            "stdout": state.get("last_stdout", "")[:500],
            "stderr": state.get("last_stderr", "")[:300],
        },
        parent_run_id=state.get("parent_run_id"),
    )

    check = assess_success(
        success_criteria=crit,
        stdout=state.get("last_stdout", ""),
        stderr=state.get("last_stderr", ""),
    )
    ok = check.get("success", False)
    why = check.get("reason", "")

    ls_end_run(
        verify_span,
        outputs={"assess": check},
        error=None if ok else "not satisfied",
    )

    if ok:
        results.append({
            "id": sid,
            "description": desc,
            "stdout": state.get("last_stdout", ""),
            "stderr": state.get("last_stderr", ""),
            "success": True,
            "reason": why,
        })

    return {
        "last_success": ok,
        "last_reason": why,
        "results_for_final": results,
    }


def route_after_verify_node(state: AgentState) -> AgentState:
    """
    Decide next route:
    - success -> next subtask or finalize
    - fail but retries left -> repair_subtask
    - fail and retries done -> record failure and move on or finalize

    We set state["route"] to "execute", "repair_subtask", or "finalize".
    """
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0)
    max_retries = state.get("max_retries", 3)
    results = state.get("results_for_final", [])
    ok = state.get("last_success", False)

    # if cursor invalid, just finalize
    if cursor < 0 or cursor >= len(subtasks):
        return {
            "route": "finalize",
            "cursor": cursor,
            "attempt": attempt,
            "results_for_final": results,
        }

    st = subtasks[cursor]
    sid = st.get("id")
    desc = st.get("description", "")

    if ok:
        # success path
        if cursor + 1 < len(subtasks):
            # proceed to next subtask
            return {
                "route": "execute",
                "cursor": cursor + 1,
                "attempt": 0,
                "results_for_final": results,
            }
        else:
            # no more subtasks
            return {
                "route": "finalize",
                "cursor": cursor,
                "attempt": attempt,
                "results_for_final": results,
            }

    # not success
    if attempt < max_retries:
        # we can retry after repair
        return {
            "route": "repair_subtask",
            "cursor": cursor,
            "attempt": attempt,  # attempt increments in execute_node
            "results_for_final": results,
        }

    # out of retries: log failure and move on
    results.append({
        "id": sid,
        "description": desc,
        "stdout": state.get("last_stdout", ""),
        "stderr": state.get("last_stderr", ""),
        "success": False,
        "reason": state.get("last_reason", ""),
    })

    if cursor + 1 < len(subtasks):
        return {
            "route": "execute",
            "cursor": cursor + 1,
            "attempt": 0,
            "results_for_final": results,
        }

    return {
        "route": "finalize",
        "cursor": cursor,
        "attempt": attempt,
        "results_for_final": results,
    }


def route_router(state: AgentState) -> str:
    """
    Tiny router fn that LangGraph calls. Must return
    'execute', 'repair_subtask', or 'finalize'.
    """
    return state.get("route", "finalize")


def repair_subtask_node(state: AgentState) -> AgentState:
    """
    Ask Gemini to fix the command for current subtask.
    Update that subtask's language/command.
    Trace 'repair_subtask' span.
    """
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0)

    if cursor < 0 or cursor >= len(subtasks):
        return {}

    st = subtasks[cursor]
    sid = st.get("id")
    desc = st.get("description", "")
    crit = st.get("success_criteria", "")
    lang = st.get("language", "cmd")
    cmd = st.get("command", "")

    repair_span = ls_start_run(
        name="repair_subtask",
        run_type="llm",
        inputs={
            "subtask_id": sid,
            "attempt": attempt,
            "old_language": lang,
            "old_command": cmd,
            "stdout": state.get("last_stdout", "")[:500],
            "stderr": state.get("last_stderr", "")[:300],
            "criteria": crit,
            "description": desc,
        },
        parent_run_id=state.get("parent_run_id"),
    )

    fix = repair_command(
        base_dir=state["base_dir"],
        subtask_desc=desc,
        success_criteria=crit,
        last_command=cmd,
        language=lang,
        stdout=state.get("last_stdout", ""),
        stderr=state.get("last_stderr", ""),
    )

    ls_end_run(
        repair_span,
        outputs={"fix": fix},
        error=None,
    )

    st["language"] = fix.get("language", lang)
    st["command"] = fix.get("command", cmd)
    subtasks[cursor] = st

    return {
        "subtasks": subtasks
    }


def finalize_node(state: AgentState) -> AgentState:
    """
    Summarize final result for the user.
    Trace 'finalize' span.
    """
    fin_span = ls_start_run(
        name="finalize",
        run_type="llm",
        inputs={
            "goal": state.get("goal", ""),
            "base_dir": state.get("base_dir", ""),
            "global_success_criteria": state.get("global_success_criteria", ""),
            "subtask_results_preview": [
                {
                    "id": r.get("id"),
                    "desc": r.get("description"),
                    "success": r.get("success"),
                } for r in state.get("results_for_final", [])
            ],
        },
        parent_run_id=state.get("parent_run_id"),
    )

    final_answer = summarize_final(
        goal=state.get("goal", ""),
        base_dir=state.get("base_dir", ""),
        global_success=state.get("global_success_criteria", ""),
        subtask_results=state.get("results_for_final", []),
    )

    ls_end_run(
        fin_span,
        outputs={"final_answer": final_answer},
        error=None,
    )

    return {
        "final_answer": final_answer
    }
