import os
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from .tools import web_search

from .state import AgentState, ls_start_run, ls_end_run
from .llm import (
    plan_tasks,
    assess_success,
    repair_command,
    summarize_final,
    ask_gemini_raw,
)

#
# ---- basic executors for cmd & python ----
#

def _run_cmd_once(command: str) -> Dict[str, Any]:
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
        return {"returncode": -1, "stdout": "", "stderr": f"Exception: {e}"}


def _run_python_once(code_str: str) -> Dict[str, Any]:
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
        return {"returncode": -1, "stdout": "", "stderr": f"Exception: {e}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


#
# ---- LLM file tool helpers ----
#

def _find_matching_file(base_dir: str, pattern: str) -> Optional[str]:
    """
    Resolve file_pattern to a single file path:
    - If absolute path and exists -> return.
    - If direct join(base_dir, pattern) exists -> return.
    - Else: walk base_dir and match by filename (case-insensitive).
    Returns first match or None.
    """
    pattern = pattern.strip().strip('"').strip("'")

    # absolute
    if os.path.isabs(pattern) and os.path.isfile(pattern):
        return pattern

    # direct under base_dir
    candidate = os.path.join(base_dir, pattern)
    if os.path.isfile(candidate):
        return candidate

    # search by name
    target_lower = os.path.basename(pattern).lower()
    for root, _dirs, files in os.walk(base_dir):
        for fn in files:
            if fn.lower() == target_lower:
                return os.path.join(root, fn)

    return None


def _read_text_file(path: str, max_chars: int = 6000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(max_chars + 1)
        if len(data) > max_chars:
            data = data[:max_chars]
        return data
    except Exception as e:
        return f"[ERROR reading text file: {e}]"


def _read_csv_head(path: str, max_lines: int = 40) -> str:
    """
    Read first N lines of a CSV or similar structured file.
    """
    try:
        lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR reading csv file: {e}]"


def _extract_pdf_text(path: str, max_chars: int = 6000) -> str:
    """
    Try to extract text from a PDF using PyPDF2 if available.
    If not installed or fails, return a hint string.
    """
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        return "[PyPDF2 not installed; cannot directly parse PDF text. File path: {}]".format(path)

    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
                if sum(len(t) for t in text) > max_chars:
                    break
        joined = "\n".join(text)
        if len(joined) > max_chars:
            joined = joined[:max_chars]
        return joined if joined.strip() else "[No extractable text from PDF {}]".format(path)
    except Exception as e:
        return f"[ERROR extracting PDF text: {e}]"


def _run_llm_file_tool(
    base_dir: str,
    description: str,
    file_pattern: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Implementation of the llm_file 'tool':
    - Locate one file matching file_pattern under base_dir.
    - Depending on extension:
        * csv/txt/log/json -> read snippet
        * pdf              -> extract snippet
        * image            -> (optional) treat as vision; here: describe via LLM with a note.
    - Call ask_gemini_raw with a constructed prompt that includes snippet.
    - Return as stdout.
    """
    path = _find_matching_file(base_dir, file_pattern)
    if not path:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"No file found for pattern '{file_pattern}' under {base_dir}",
        }

    ext = os.path.splitext(path)[1].lower()
    snippet = ""

    if ext in (".csv", ".tsv"):
        snippet = _read_csv_head(path)
    elif ext in (".txt", ".log", ".md", ".json"):
        snippet = _read_text_file(path)
    elif ext in (".pdf",):
        snippet = _extract_pdf_text(path)
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        # For simplicity we do a text-only wrapper.
        # If you enable Gemini vision with inline_data, you can enhance this.
        snippet = "[Image file at {}. You may describe or infer based on its content if vision is enabled.]".format(path)
    else:
        # generic text read attempt
        snippet = _read_text_file(path)

    # Build LLM prompt
    prompt = f"""
You are an assistant with access to the contents of a local file.

User goal / subtask:
{description}

User instruction for this file:
{user_prompt}

File path:
{path}

Here is a snippet / extracted content from the file:
----------------
{snippet}
----------------

Based ONLY on this file content and the instructions, provide a helpful answer.
If the snippet looks truncated, still answer with whatever is visible.

Your answer will be captured as stdout for an automated agent.
Do NOT include JSON; just answer in natural language.
"""

    answer = ask_gemini_raw(prompt)

    return {
        "returncode": 0,
        "stdout": f"LLM_FILE_TOOL RESULT (from {os.path.basename(path)}):\n{answer.strip()}",
        "stderr": "",
    }


def _exec_once(
    language: str,
    command: str,
    base_dir: str,
    subtask: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Dispatch subtask execution:
    - cmd       -> shell/PowerShell
    - python    -> temp script
    - llm_file  -> local file + LLM analysis
    """
    if language == "python":
        return _run_python_once(command)
    if language == "llm_file":
        file_pattern = subtask.get("file_pattern", "")
        user_prompt = subtask.get("prompt", "") or subtask.get("description", "")
        return _run_llm_file_tool(
            base_dir=base_dir,
            description=subtask.get("description", ""),
            file_pattern=file_pattern,
            user_prompt=user_prompt,
        )
    if language == "web_search":
        query = subtask.get("query", command or subtask.get("description", ""))
        context = subtask.get("description", "")
        summary = web_search(query=query, context=context)
        return {"returncode": 0, "stdout": summary, "stderr": ""}
    # default: cmd
    return _run_cmd_once(command)


#
# ---- LangGraph node implementations ----
#

def plan_node(state: AgentState) -> AgentState:
    parent_id = state.get("parent_run_id")

    span = ls_start_run(
        name="plan_tasks",
        run_type="llm",
        inputs={"goal": state["goal"], "base_dir": state["base_dir"]},
        parent_run_id=parent_id,
    )

    plan = plan_tasks(
        chat_history=state["chat_history"],
        goal=state["goal"],
        base_dir=state["base_dir"],
    )

    ls_end_run(span, outputs={"plan": plan}, error=None)

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
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0) + 1
    base_dir = state.get("base_dir", os.getcwd())

    if cursor < 0 or cursor >= len(subtasks):
        return {"attempt": attempt, "last_stdout": "", "last_stderr": ""}

    st = subtasks[cursor]
    lang = st.get("language", "cmd")
    cmd = st.get("command", "")
    desc = st.get("description", "")

    span = ls_start_run(
        name="execute",
        run_type="tool",
        inputs={
            "subtask_id": st.get("id"),
            "attempt": attempt,
            "description": desc,
            "language": lang,
            "command": cmd,
            "file_pattern": st.get("file_pattern", ""),
            "prompt": st.get("prompt", ""),
        },
        parent_run_id=state.get("parent_run_id"),
    )

    res = _exec_once(lang, cmd, base_dir=base_dir, subtask=st)

    ls_end_run(
        span,
        outputs={
            "returncode": res.get("returncode"),
            "stdout": res.get("stdout", "")[:1200],
            "stderr": res.get("stderr", "")[:400],
        },
        error=None,
    )

    return {
        "attempt": attempt,
        "last_stdout": res.get("stdout", ""),
        "last_stderr": res.get("stderr", ""),
    }


def verify_node(state: AgentState) -> AgentState:
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

    span = ls_start_run(
        name="verify",
        run_type="llm",
        inputs={
            "subtask_id": sid,
            "attempt": attempt,
            "description": desc,
            "success_criteria": crit,
            "stdout": state.get("last_stdout", "")[:600],
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
        span,
        outputs={"assess": check},
        error=None if ok else "not satisfied",
    )

    # record success immediately
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
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0)
    max_retries = state.get("max_retries", 3)
    results = state.get("results_for_final", [])
    ok = state.get("last_success", False)

    if cursor < 0 or cursor >= len(subtasks):
        return {"route": "finalize", "cursor": cursor, "attempt": attempt,
                "results_for_final": results}

    st = subtasks[cursor]
    sid = st.get("id")
    desc = st.get("description", "")

    if ok:
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

    # fail
    if attempt < max_retries and st.get("language") in ("cmd", "python"):
        # only cmd/python tasks go through repair loop
        return {
            "route": "repair_subtask",
            "cursor": cursor,
            "attempt": attempt,
            "results_for_final": results,
        }

    # out of retries or llm_file failure => log failure and move on/finalize
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
    return state.get("route", "finalize")


def repair_subtask_node(state: AgentState) -> AgentState:
    subtasks = state.get("subtasks", [])
    cursor = state.get("cursor", 0)
    attempt = state.get("attempt", 0)

    if cursor < 0 or cursor >= len(subtasks):
        return {}

    st = subtasks[cursor]

    # Only repair cmd/python; llm_file is conceptual
    if st.get("language") not in ("cmd", "python"):
        return {}

    sid = st.get("id")
    desc = st.get("description", "")
    crit = st.get("success_criteria", "")
    lang = st.get("language", "cmd")
    cmd = st.get("command", "")

    span = ls_start_run(
        name="repair_subtask",
        run_type="llm",
        inputs={
            "subtask_id": sid,
            "attempt": attempt,
            "old_language": lang,
            "old_command": cmd,
            "stdout": state.get("last_stdout", "")[:600],
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

    ls_end_run(span, outputs={"fix": fix}, error=None)

    st["language"] = fix.get("language", lang)
    st["command"] = fix.get("command", cmd)
    subtasks[cursor] = st

    return {"subtasks": subtasks}


def finalize_node(state: AgentState) -> AgentState:
    span = ls_start_run(
        name="finalize",
        run_type="llm",
        inputs={
            "goal": state.get("goal", ""),
            "base_dir": state.get("base_dir", ""),
            "global_success_criteria": state.get("global_success_criteria", ""),
        },
        parent_run_id=state.get("parent_run_id"),
    )

    final_answer = summarize_final(
        goal=state.get("goal", ""),
        base_dir=state.get("base_dir", ""),
        global_success=state.get("global_success_criteria", ""),
        subtask_results=state.get("results_for_final", []),
    )

    ls_end_run(span, outputs={"final_answer": final_answer}, error=None)

    return {"final_answer": final_answer}
