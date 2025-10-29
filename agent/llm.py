import os
import json
from typing import Any, Dict, List
from google import genai



def _get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return genai.Client()
    try:
        return genai.Client(api_key=api_key)
    except TypeError:
        return genai.Client()


def ask_gemini_raw(prompt: str) -> str:
    try:
        client = _get_client()
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        try:
            return resp.text
        except Exception:
            return str(resp)
    except Exception as e:
        return f"[LLM call failed: {e}]"



def _try_parse_json(raw: str, fallback: Any) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    start_brace = raw.find("{")
    start_brack = raw.find("[")
    starts = [i for i in (start_brace, start_brack) if i != -1]
    if starts:
        start = min(starts)
        end_curly = raw.rfind("}")
        end_brack = raw.rfind("]")
        ends = [i for i in (end_curly, end_brack) if i != -1]
        if ends:
            end = max(ends) + 1
            snippet = raw[start:end]
            try:
                return json.loads(snippet)
            except Exception:
                pass

    return fallback


def _format_chat_history(history: List[Dict[str, str]], limit: int = 8) -> str:
    """
    Make a short transcript for planner.
    history is like:
      [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
    """
    take = history[-limit:]
    lines = []
    for msg in take:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)



def plan_tasks(
    chat_history: List[Dict[str, str]],
    goal: str,
    base_dir: str
) -> Dict[str, Any]:
    """
    Ask Gemini to break the user goal into ordered subtasks.

    Expected ONLY valid JSON:
    {
      "global_success_criteria": "...",
      "subtasks": [
        {
          "id": 1,
          "description": "what this subtask does",
          "success_criteria": "how we know this subtask succeeded",
          "language": "cmd" | "python",
          "command": "actual command or python script"
        },
        ...
      ]
    }

    - Use base_dir ("AGENT_BASE_DIR") as default search root unless user gave
      an absolute path.
    - For python subtasks: MUST be full, self-contained, fast scripts using ONLY
      Python standard library (os, sys, csv, glob, pathlib, etc.).
      NO pandas, NO pip install.
      OK: open file, walk dirs, print first 5 lines.
    - For cmd subtasks: MUST be one Windows command or PowerShell command
      (no && chains).
    - success_criteria must be about what's visible in stdout/stderr.

    We'll run subtasks in order. For each subtask:
       execute -> verify -> if fail -> repair -> retry (up to max_retries)
    """

    chat_context = _format_chat_history(chat_history)

    planner_prompt = f"""
You are the PLANNER for a local Windows automation agent.

AGENT_BASE_DIR = "{base_dir}"

User wants to achieve a goal using this computer.
You will output ONLY VALID JSON (no commentary outside JSON) with this shape:

{{
  "global_success_criteria": "string explaining when the WHOLE goal is done",
  "subtasks": [
    {{
      "id": 1,
      "description": "natural language meaning of the subtask",
      "success_criteria": "stdout/stderr condition that proves success",
      "language": "cmd" | "python",
      "command": "command or FULL python script body"
    }},
    {{
      "id": 2,
      "description": "...",
      "success_criteria": "...",
      "language": "cmd" | "python",
      "command": "..."
    }}
  ]
}}

RULES:
1. If user wants info about files, folders, CSVs, etc., assume those are under
   AGENT_BASE_DIR unless user mentions an absolute path elsewhere.
   Use quoted Windows paths if they contain spaces,
   e.g. "C:\\Users\\Swastik R\\Downloads\\Data Formats".
2. If language == "cmd":
   - Provide ONE PowerShell/cmd command only.
   - No "&&" chaining.
3. If language == "python":
   - Provide a COMPLETE python script body that we can write to temp.py and run.
   - Standard library only (os, sys, csv, io, glob, pathlib, etc.).
   - Must run quickly (under ~2 seconds).
   - If printing first 5 lines of a CSV, do something like:
        import os
        target = r"...absolute\\path\\file.csv"
        if os.path.isfile(target):
            print("FILE_PATH:", target)
            with open(target, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= 5: break
                    print(line.rstrip())
        else:
            print("FILE_NOT_FOUND", target)
4. success_criteria must describe the expected stdout/stderr contents.
   Example:
   "stdout must include at least 5 CSV lines and the absolute path of the file."
   "stdout must list all .pdf files in the folder, one per line."
   "stdout must include CPU load percentage and RAM in GB."

We'll run each subtask in sequence.
After each we capture stdout/stderr and check success_criteria.
If it fails, we will ask you (in repair mode) to fix the command and retry.

CHAT HISTORY:
{chat_context}

USER GOAL:
"{goal}"

Return ONLY the JSON object. No English outside the JSON.
"""

    fallback = {
        "global_success_criteria": "We produced useful output.",
        "subtasks": [
            {
                "id": 1,
                "description": goal,
                "success_criteria": "Some output was printed.",
                "language": "cmd",
                "command": f'echo {goal}',
            }
        ],
    }

    raw = ask_gemini_raw(planner_prompt)
    data = _try_parse_json(raw, fallback=fallback)

    if not isinstance(data, dict):
        data = fallback

    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list) or not subtasks:
        data = fallback
        subtasks = fallback["subtasks"]

    cleaned = []
    for st in subtasks:
        if not isinstance(st, dict):
            continue
        sid = st.get("id", None)
        desc = (st.get("description", "") or "").strip()
        crit = (st.get("success_criteria", "") or "").strip()
        lang = (st.get("language", "") or "").strip().lower()
        cmd = (st.get("command", "") or "").strip()

        if lang not in ("cmd", "python"):
            lang = "cmd"
        if not cmd:
            continue

        cleaned.append({
            "id": sid,
            "description": desc,
            "success_criteria": crit,
            "language": lang,
            "command": cmd,
        })

    if not cleaned:
        cleaned = fallback["subtasks"]

    return {
        "global_success_criteria": data.get(
            "global_success_criteria",
            "We produced useful output."
        ),
        "subtasks": cleaned,
    }




def assess_success(
    success_criteria: str,
    stdout: str,
    stderr: str
) -> Dict[str, Any]:
    """
    Ask Gemini if success_criteria is satisfied.
    Respond ONLY JSON:
    {
      "success": true/false,
      "reason": "short reason"
    }
    """

    prompt = f"""
You are the SUCCESS CHECKER.

SUBTASK SUCCESS CRITERIA:
{success_criteria}

COMMAND STDOUT:
{stdout[:2000]}

COMMAND STDERR:
{stderr[:800]}

Reply ONLY valid JSON:
{{
  "success": true/false,
  "reason": "short explanation"
}}
"""

    raw = ask_gemini_raw(prompt)
    fallback = {"success": False, "reason": "fallback: parse failed"}
    data = _try_parse_json(raw, fallback=fallback)

    if not isinstance(data, dict):
        data = fallback
    if "success" not in data:
        data["success"] = False
    if "reason" not in data:
        data["reason"] = ""
    return data



def repair_command(
    base_dir: str,
    subtask_desc: str,
    success_criteria: str,
    last_command: str,
    language: str,
    stdout: str,
    stderr: str,
) -> Dict[str, Any]:
    """
    Ask Gemini to fix a failing subtask's command.
    Return ONLY JSON:
    {
      "language": "cmd" | "python",
      "command": "..."
    }
    """

    prompt = f"""
You are the COMMAND REPAIR AGENT.

AGENT_BASE_DIR = "{base_dir}"

SUBTASK DESCRIPTION:
{subtask_desc}

SUBTASK SUCCESS CRITERIA:
{success_criteria}

We attempted this ({language}):
{last_command}

Result:
STDOUT:
{stdout[:2000]}
STDERR:
{stderr[:800]}

We did NOT meet success_criteria.

Fix it. Return ONLY valid JSON:
{{
  "language": "cmd" | "python",
  "command": "..."
}}

Rules:
- If "cmd": return ONE PowerShell/cmd command (no &&).
- If "python": return a COMPLETE python script body we can run directly.
- Use ONLY python stdlib. No pandas, no pip install.
- Must run quickly (<2 seconds).
- Quote Windows paths with spaces, e.g. "C:\\Users\\Swastik R\\Downloads\\Data Formats".
- Reuse discovered absolute file paths from stdout if helpful.

No commentary outside the JSON.
"""

    raw = ask_gemini_raw(prompt)
    fallback = {
        "language": language,
        "command": last_command,
    }

    data = _try_parse_json(raw, fallback=fallback)
    if not isinstance(data, dict):
        data = fallback

    new_lang = (data.get("language", language) or "").strip().lower()
    if new_lang not in ("cmd", "python"):
        new_lang = "cmd"

    new_cmd = (data.get("command", last_command) or "").strip()

    return {
        "language": new_lang,
        "command": new_cmd,
    }




def summarize_final(
    goal: str,
    base_dir: str,
    global_success: str,
    subtask_results: List[Dict[str, Any]]
) -> str:
    """
    Ask Gemini to generate the final answer for the user, describing what
    happened, what paths we found, results from CSV previews, system stats, etc.
    """

    compact = []
    for r in subtask_results:
        compact.append({
            "id": r.get("id"),
            "description": r.get("description", ""),
            "success": r.get("success", False),
            "reason": r.get("reason", ""),
            "stdout": (r.get("stdout", "")[:1200]),
            "stderr": (r.get("stderr", "")[:400]),
        })
    results_json_str = json.dumps(compact, indent=2)

    prompt = f"""
You are the FINAL RESPONSE WRITER.

AGENT_BASE_DIR = "{base_dir}"

USER GOAL:
{goal}

GLOBAL SUCCESS CRITERIA FOR WHOLE GOAL:
{global_success}

SUBTASK RESULTS:
{results_json_str}

Write a final message for the user:
- Summarize what we actually did on the machine.
- Mention important file paths found.
- If we printed CSV rows, summarize them or include them briefly.
- Summarize system info if collected (CPU usage, RAM, disk).
- Be honest about any failures after retries.
- Be clear and helpful.
- Do NOT include internal debugging or stack traces.

Return plain English only.
"""

    return ask_gemini_raw(prompt).strip()
