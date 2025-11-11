import os
import json
from typing import Any, Dict, List
from google import genai



#
# --- Gemini client helpers ---
#

def _get_client():
    """
    Build Gemini client from GEMINI_API_KEY in .env.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return genai.Client()
    try:
        return genai.Client(api_key=api_key)
    except TypeError:
        # Some SDKs auto-read env, ignore mismatch
        return genai.Client()


def ask_gemini_raw(prompt: str) -> str:
    """
    Send a pure-text prompt to Gemini 2.5 Flash and get response.text.
    """
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


#
# --- JSON parsing helper ---
#

def _try_parse_json(raw: str, fallback: Any) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try slice {...} or [...]
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
    take = history[-limit:]
    lines = []
    for msg in take:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


#
# --- Planner LLM ---
#

def plan_tasks(
    chat_history: List[Dict[str, str]],
    goal: str,
    base_dir: str
) -> Dict[str, Any]:
    """
    Planner can now emit four kinds of subtasks:

    1) Shell command:
       {
         "id": 1,
         "description": "...",
         "success_criteria": "...",
         "language": "cmd",
         "command": "dir ..."
       }

    2) Python script:
       {
         "id": 2,
         "description": "...",
         "success_criteria": "...",
         "language": "python",
         "command": "import os\n..."
       }

    3) LLM file tool (NEW):
       {
         "id": 3,
         "description": "Have the LLM read/summarize heart.csv",
         "success_criteria": "stdout must contain an answer based on heart.csv contents.",
         "language": "llm_file",
         "file_pattern": "heart.csv",
         "prompt": "Read heart.csv and show the first 5 rows and column names."
       }

    For llm_file:
      - file_pattern: relative name or glob-like hint under AGENT_BASE_DIR,
                      or an absolute path.
      - prompt: instruction/question for that file.
      The node will:
        - locate the file
        - read or extract content (CSV/TXT/PDF)
        - call Gemini with the prompt + content snippet
        - print the LLM answer as stdout.
    
    4) Web search tool (NEW):
        {
         "id": 4,
         "description": "Find the latest information about Tesla earnings",
         "success_criteria": "stdout must list relevant links and short summaries.",
         "language": "web_search",
         "query": "latest Tesla earnings report"
       }

       For web_search:
         - query: the search query string to use.
         The node will:
           - perform a DuckDuckGo or Tavily search
           - retrieve up to N results (default 5)
           - summarize and format each with title, snippet, and URL
           - print the summarized results as stdout.
    """

    chat_context = _format_chat_history(chat_history)

    planner_prompt = f"""
You are the PLANNER for a local Windows + LLM agent.

AGENT_BASE_DIR = "{base_dir}"

You receive a USER_GOAL and recent chat history.
You must break the goal into 1..6 ordered subtasks.

OUTPUT FORMAT (ONLY VALID JSON):

{{
  "global_success_criteria": "string",
  "subtasks": [
    {{
      "id": 1,
      "description": "string",
      "success_criteria": "string",
      "language": "cmd" | "python" | "llm_file" | "web_search",
      "command": "string (for cmd/python only)",
      "file_pattern": "optional, for llm_file",
      "prompt": "optional, for llm_file",
      "query": "optional, for web_search"
    }},
    ...
  ]
}}

INTERPRETATION:

1) language="cmd":
   - Provide ONE Windows cmd/PowerShell command (no &&).
   - Use quoted absolute paths if they contain spaces.
   - Use AGENT_BASE_DIR as default root unless user gave an absolute path.

2) language="python":
   - Provide a FULL Python script body.
   - Standard library only (os, sys, csv, glob, pathlib, etc.).
   - No external installs.
   - Must run quickly.
   - Good for structured reading (e.g., print first 5 lines of a CSV).

3) language="llm_file" (NEW):
   - Use when the user asks:
       * "summarize this PDF"
       * "analyze this CSV"
       * "what's in this image"
       * "explain the content of ..."
   - Fields:
       * file_pattern: required. A filename or simple pattern,
         relative to AGENT_BASE_DIR unless absolute path is given.
         e.g. "heart.csv", "FML Mid.pdf", "data/*.csv".
       * prompt: required. Natural language instruction for analyzing the file.
   - Do NOT put shell/python code in "command" for llm_file.
   - The runtime will:
       * find the file(s),
       * read text (CSV/TXT/PDF) or image bytes,
       * call the LLM with (prompt + file content),
       * return that answer via stdout.

4) language="web_search" (NEW):
   - Use when the user asks for current or online information,
     e.g. "latest AI news", "recent Tesla earnings", "top YouTube trends".
   - Fields:
       * query: required. The web search query string.
   - The runtime will:
       * perform a Tavily or DuckDuckGo search (max 5 results)
       * summarize results for relevance
       * return concise findings via stdout.
   - Use this when the needed data is *not* available locally.
       
SUCCESS CRITERIA:
- For every subtask, success_criteria must reference what we should see in stdout/stderr.
- Examples:
    "stdout must list at least one matching path"
    "stdout must contain 5 CSV rows and the file path"
    "stdout must be a summary of the PDF sections"
    "stdout must list relevant links and summaries from online sources"

CHAT HISTORY:
{chat_context}

USER_GOAL:
"{goal}"

Return ONLY the JSON object, no extra commentary.
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
        file_pattern = (st.get("file_pattern", "") or "").strip()
        llm_prompt = (st.get("prompt", "") or "").strip()
        query = (st.get("query", "") or "").strip()

        # Validate language
        if lang not in ("cmd", "python", "llm_file", "web_search"):
            lang = "cmd"

        # Validate fields
        if lang in ("cmd", "python") and not cmd:
            continue
        if lang == "llm_file" and (not file_pattern or not llm_prompt):
            continue
        if lang == "web_search" and not query:
            continue

        # Build clean entry
        entry = {
            "id": sid,
            "description": desc,
            "success_criteria": crit,
            "language": lang,
        }

        if lang in ("cmd", "python"):
            entry["command"] = cmd
        elif lang == "llm_file":
            entry["file_pattern"] = file_pattern
            entry["prompt"] = llm_prompt
        elif lang == "web_search":
            entry["query"] = query

        cleaned.append(entry)

    if not cleaned:
        cleaned = fallback["subtasks"]

    return {
        "global_success_criteria": data.get(
            "global_success_criteria",
            "We produced useful output."
        ),
        "subtasks": cleaned,
    }


#
# --- Verifier LLM ---
#

def assess_success(
    success_criteria: str,
    stdout: str,
    stderr: str
) -> Dict[str, Any]:
    """
    Ask Gemini if success_criteria is satisfied.

    Returns ONLY:
    {
      "success": true/false,
      "reason": "..."
    }
    """

    prompt = f"""
You are the SUCCESS CHECKER.

SUCCESS CRITERIA:
{success_criteria}

STDOUT:
{stdout[:2000]}

STDERR:
{stderr[:800]}

Is the criteria satisfied?

Return ONLY JSON:
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


#
# --- Repair LLM ---
#

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
    Only used for cmd/python subtasks.
    llm_file subtasks are conceptual; if they fail, planner should fix them in next loop.
    """

    prompt = f"""
You are the COMMAND REPAIR AGENT.

AGENT_BASE_DIR = "{base_dir}"

SUBTASK:
{subtask_desc}

SUCCESS CRITERIA:
{success_criteria}

We attempted this ({language}):
{last_command}

Result:
STDOUT:
{stdout[:1500]}
STDERR:
{stderr[:800]}

We did NOT meet the criteria.

Return ONLY JSON:
{{
  "language": "cmd" | "python",
  "command": "..."
}}

Rules:
- Keep it to ONE cmd or ONE full python script.
- Python = stdlib only, no installs.
- Use absolute, quoted Windows paths when needed.
"""

    raw = ask_gemini_raw(prompt)
    fallback = {"language": language, "command": last_command}
    data = _try_parse_json(raw, fallback=fallback)

    if not isinstance(data, dict):
        data = fallback

    new_lang = (data.get("language", language) or "").strip().lower()
    if new_lang not in ("cmd", "python"):
        new_lang = "cmd"
    new_cmd = (data.get("command", last_command) or "").strip()

    return {"language": new_lang, "command": new_cmd}


#
# --- Final summarizer LLM ---
#

def summarize_final(
    goal: str,
    base_dir: str,
    global_success: str,
    subtask_results: List[Dict[str, Any]]
) -> str:
    """
    Build final natural-language answer for the user.
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
    summary_json = json.dumps(compact, indent=2)

    prompt = f"""
You are the FINAL RESPONSE WRITER for a local computer assistant.

AGENT_BASE_DIR = "{base_dir}"

USER GOAL:
{goal}

GLOBAL SUCCESS CRITERIA:
{global_success}

SUBTASK RESULTS:
{summary_json}

Write a clear final response:
- Explain what the agent actually did.
- Mention important file paths.
- For CSV/PDF analysis, summarize key findings or show short excerpts.
- For images, describe what was inferred.
- Include system info if relevant.
- Be honest about any failures.
- No debug traces, no JSON, just a nice explanation.

Return plain text only.
"""

    return ask_gemini_raw(prompt).strip()
