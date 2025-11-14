# agent/tools.py
import os
import json
import subprocess
import psutil
import platform
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
from tavily import TavilyClient
from .llm import ask_gemini_raw

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# Optional deps
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pandas as pd
except ImportError:
    pd = None

# For Windows-specific operations
if platform.system() == 'Windows':
    try:
        import wmi
        import screen_brightness_control as sbc
        HAS_WINDOWS_TOOLS = True
    except ImportError:
        HAS_WINDOWS_TOOLS = False
else:
    HAS_WINDOWS_TOOLS = False


# --------- low-level helpers ---------

def _run_shell_cmd(command: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Run a single command string via Windows shell.
    We don't chain with &&; Planner must emit single commands.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        return {
            "ok": (proc.returncode == 0),
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Exception while running command: {e}",
        }


def _human_size(bytes_val: int) -> str:
    # tiny helper for pretty file sizes if we need it later
    if bytes_val < 1024:
        return f"{bytes_val} B"
    kb = bytes_val / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.1f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = {
            "cpu_usage": f"{cpu_percent}%",
            "ram_total": _human_size(memory.total),
            "ram_available": _human_size(memory.available),
            "ram_used_percent": f"{memory.percent}%",
            "disk_total": _human_size(disk.total),
            "disk_free": _human_size(disk.free),
            "disk_used_percent": f"{disk.percent}%"
        }
        
        if HAS_WINDOWS_TOOLS:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    info["battery_percent"] = f"{battery.percent}%"
                    info["battery_charging"] = battery.power_plugged
                
                info["screen_brightness"] = sbc.get_brightness()[0]
            except Exception:
                pass
                
        return {
            "ok": True,
            "stdout": json.dumps(info, indent=2),
            "stderr": ""
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error getting system info: {str(e)}"
        }

def get_running_processes() -> Dict[str, Any]:
    """Get list of running processes with CPU and memory usage."""
    try:
        process_list = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                process_list.append({
                    "pid": pinfo["pid"],
                    "name": pinfo["name"],
                    "cpu_percent": f"{pinfo['cpu_percent']:.1f}%",
                    "memory_percent": f"{pinfo['memory_percent']:.1f}%"
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # Sort by CPU usage
        process_list.sort(key=lambda x: float(x["cpu_percent"].rstrip('%')), reverse=True)
        
        return {
            "ok": True,
            "stdout": json.dumps(process_list[:10], indent=2),  # Top 10 processes
            "stderr": ""
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error getting process list: {str(e)}"
        }


# --------- py action implementations ---------

def _py_search_folder(term: str, start_root: str) -> Dict[str, Any]:
    """
    Walk start_root looking for folder names containing `term` (case-insensitive).
    """
    matches = []
    term_lower = term.lower()
    for root, dirs, files in os.walk(start_root):
        base = os.path.basename(root)
        if term_lower in base.lower():
            matches.append(root)

    out_lines = [
        f"Folder search term: {term}",
        "Matches:",
    ]
    if matches:
        out_lines.extend(matches)
    else:
        out_lines.append("(no matches)")

    stdout = "\n".join(out_lines)
    return {
        "ok": True,
        "stdout": stdout,
        "stderr": "",
    }


def analyze_image(image_path: str) -> Dict[str, Any]:
    """Analyze an image file and return its properties."""
    if not Image:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "Pillow library not installed. Install with: pip install Pillow"
        }
        
    try:
        with Image.open(image_path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "size": f"{img.size[0]}x{img.size[1]}",
                "width": img.size[0],
                "height": img.size[1]
            }
            
            return {
                "ok": True,
                "stdout": json.dumps(info, indent=2),
                "stderr": ""
            }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error analyzing image: {str(e)}"
        }

def summarize_pdf(pdf_path: str, max_pages: int = 3) -> Dict[str, Any]:
    """Extract and summarize the first few pages of a PDF."""
    if not PyPDF2:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "PyPDF2 library not installed. Install with: pip install PyPDF2"
        }
        
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            pages_to_read = min(max_pages, num_pages)
            
            text_content = []
            for i in range(pages_to_read):
                page = reader.pages[i]
                text_content.append(page.extract_text())
            
            summary = "\n".join(text_content)
            
            # Use Gemini to generate a concise summary
            prompt = f"Please provide a concise bullet-point summary of this text: {summary[:2000]}..."
            summary_result = ask_gemini_raw(prompt)
            
            info = {
                "total_pages": num_pages,
                "pages_read": pages_to_read,
                "summary": summary_result
            }
            
            return {
                "ok": True,
                "stdout": json.dumps(info, indent=2),
                "stderr": ""
            }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error summarizing PDF: {str(e)}"
        }

def analyze_csv(csv_path: str) -> Dict[str, Any]:
    """Provide quick statistics about a CSV file."""
    if not pd:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "Pandas library not installed. Install with: pip install pandas"
        }
        
    try:
        df = pd.read_csv(csv_path)
        info = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview": df.head(5).to_dict(orient='records')
        }
        
        return {
            "ok": True,
            "stdout": json.dumps(info, indent=2),
            "stderr": ""
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error analyzing CSV: {str(e)}"
        }

def _py_list_dir(path: str) -> Dict[str, Any]:
    """
    List immediate children of `path`.
    Return both human-readable listing and a machine-readable block:
    FILES_JSON_BEGIN ... FILES_JSON_END
    We'll also count total size of regular files only (non-directories).
    """
    entries: List[Dict[str, Any]] = []
    lines: List[str] = []
    total_size = 0

    try:
        with os.scandir(path) as it:
            for entry in it:
                abs_path = os.path.join(path, entry.name)
                if entry.is_dir():
                    size_bytes = 0
                else:
                    try:
                        size_bytes = os.path.getsize(abs_path)
                        total_size += size_bytes
                    except Exception:
                        size_bytes = -1

                ext = os.path.splitext(entry.name)[1]

                entries.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size_bytes": size_bytes,
                    "ext": ext,
                    "abs_path": abs_path,
                })

                if entry.is_dir():
                    lines.append(f"<DIR> {entry.name}")
                else:
                    lines.append(f"{size_bytes} bytes {entry.name}")

        listing_json_str = json.dumps(entries)

        stdout = (
            f"Contents of {path}:\n"
            + "\n".join(lines)
            + f"\nItems: {len(entries)}\n"
            + f"TotalSizeBytes(non-recursive): {total_size}\n"
            + "FILES_JSON_BEGIN\n"
            + listing_json_str
            + "\nFILES_JSON_END\n"
        )
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error listing directory {path}: {e}",
        }


def _py_analyze_listing(listing_json: str, task: str, value: str = "") -> Dict[str, Any]:
    """
    Analyze a cached directory listing. listing_json is a JSON array of file dicts:
    [
      {"name": "...", "is_dir": false, "size_bytes":1234,
       "ext":".pdf", "abs_path":"C:\\..."},
      ...
    ]

    task == "filter_ext":
        value=".pdf" or ".py" etc.
        We'll output only those files (ignore directories).
    task == "largest_file":
        find max size_bytes among non-directories
    """
    try:
        data = json.loads(listing_json)
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Could not parse listing_json: {e}",
        }

    if not isinstance(data, list):
        return {
            "ok": False,
            "stdout": "",
            "stderr": "listing_json is not a list",
        }

    if task == "filter_ext":
        target_ext = value.lower().strip()
        matches = []
        for item in data:
            if item.get("is_dir"):
                continue
            ext = str(item.get("ext", "")).lower()
            if ext == target_ext:
                matches.append(item)

        if not matches:
            stdout = (
                f"No files with extension {target_ext} found."
            )
        else:
            lines = [f"Filtered files with extension {target_ext}:"]
            for m in matches:
                nm = m.get("name", "?")
                sz = m.get("size_bytes", 0)
                pth = m.get("abs_path", "?")
                lines.append(
                    f"- {nm} ({sz} bytes) - {pth}"
                )
            lines.append(f"Count: {len(matches)}")
            stdout = "\n".join(lines)

        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }

    elif task == "largest_file":
        best = None
        for item in data:
            if item.get("is_dir"):
                continue
            size_b = item.get("size_bytes", -1)
            if best is None or size_b > best.get("size_bytes", -1):
                best = item

        if best is None:
            stdout = "No regular files found to compare sizes."
        else:
            stdout = (
                "Largest file:\n"
                f"Name: {best.get('name','?')}\n"
                f"SizeBytes: {best.get('size_bytes','?')}\n"
                f"Path: {best.get('abs_path','?')}\n"
            )

        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }

    else:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Unknown analyze_listing task: {task}",
        }


def _py_count_folders(path: str) -> Dict[str, Any]:
    """
    Count direct subdirectories in the given path.
    """
    try:
        cnt = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    cnt += 1
        stdout = f"FolderCount: {cnt} Path: {path}"
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error counting folders in {path}: {e}",
        }


def _py_create_dir(path: str) -> Dict[str, Any]:
    """
    Create a folder (if already exists that's fine).
    """
    try:
        os.makedirs(path, exist_ok=True)
        stdout = f"Directory created or already exists: {path}"
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error creating dir {path}: {e}",
        }


def _py_write_python_file(path: str, purpose: str) -> Dict[str, Any]:
    """
    Generate starter python code using Gemini, then write it to disk.
    """
    gen_prompt = (
        "Write a short Python script. Purpose:\n"
        f"{purpose}\n\n"
        "Rules:\n"
        "- Use only Python standard library if possible.\n"
        "- Add helpful comments.\n"
        "- Do not include backticks.\n"
    )
    code_body = ask_gemini_raw(gen_prompt).strip()

    try:
        # ensure parent dir exists
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(code_body)

        stdout = f"Python file written at {path}\n"
        stdout += "File preview (first ~200 chars):\n"
        stdout += code_body[:200]
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error writing python file {path}: {e}",
        }


def _py_summarize_pdf(path: str) -> Dict[str, Any]:
    """
    Extract text from the first few pages of a PDF and ask Gemini for summary.
    """
    if PyPDF2 is None:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "PyPDF2 not installed, cannot summarize PDF.",
        }

    try:
        reader = PyPDF2.PdfReader(path)
        text_chunks = []
        max_pages = min(5, len(reader.pages))
        for i in range(max_pages):
            try:
                text_chunks.append(reader.pages[i].extract_text() or "")
            except Exception:
                text_chunks.append("")
        raw_text = "\n".join(text_chunks)
        raw_text = raw_text[:8000]  # clip so prompt not giant

        sum_prompt = (
            "Summarize this PDF content in 3-5 bullet points in plain English:\n\n"
            f"{raw_text}\n\n"
            "Focus on the main topics, purpose, and conclusions."
        )
        summary = ask_gemini_raw(sum_prompt).strip()
        stdout = f"PDF Summary:\n{summary}"
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error summarizing PDF {path}: {e}",
        }


def _py_inspect_image_basic(path: str) -> Dict[str, Any]:
    """
    Simple image info without vision model.
    """
    if Image is None:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "Pillow not installed, cannot inspect image.",
        }
    try:
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
        stdout = (
            f"Image info: Path: {path} Size: {w}x{h}; Mode: {mode}"
        )
        return {
            "ok": True,
            "stdout": stdout,
            "stderr": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error inspecting image {path}: {e}",
        }


def run_py_action(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch table for tool="py".
    """
    if action == "search_folder":
        term = args.get("term", "")
        rootp = args.get("start_root", "C:\\")
        return _py_search_folder(term, rootp)

    if action == "list_dir":
        path = args.get("path", "")
        return _py_list_dir(path)

    if action == "analyze_listing":
        listing_json = args.get("listing_json", "")
        task = args.get("task", "")
        value = args.get("value", "")
        return _py_analyze_listing(listing_json, task, value)

    if action == "count_folders":
        path = args.get("path", "")
        return _py_count_folders(path)

    if action == "create_dir":
        path = args.get("path", "")
        return _py_create_dir(path)

    if action == "write_python_file":
        path = args.get("path", "")
        purpose = args.get("purpose", "")
        return _py_write_python_file(path, purpose)

    if action == "summarize_pdf":
        path = args.get("path", "")
        return _py_summarize_pdf(path)

    if action == "inspect_image_basic":
        path = args.get("path", "")
        return _py_inspect_image_basic(path)

    return {
        "ok": False,
        "stdout": "",
        "stderr": f"Unknown py action: {action}",
    }


def run_tool_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified executor.
    step = {
      "tool": "cmd" | "py",
      ...
    }
    """
    tool = step.get("tool", "")
    if tool == "cmd":
        command = step.get("command", "")
        return _run_shell_cmd(command)
    elif tool == "py":
        action = step.get("action", "")
        args = step.get("args", {}) or {}
        return run_py_action(action, args)
    else:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Unknown tool type: {tool}",
        }

def _summarize_search_results(
    query: str,
    context: str,
    snippets: List[str],
    source_links: List[str],
) -> str:
    if not snippets:
        return f"No results found for '{query}'."

    joined_snippets = "\n".join(snippets)
    search_summary_prompt = f"""
You are a summarization agent with access to multiple web search results.

USER GOAL / CONTEXT:
{context}

QUERY:
{query}

Here are the top results:
----------------
{joined_snippets}
----------------

Please synthesize a concise, well-organized summary:
- Include the most relevant information only.
- Mention key findings or facts.
- Add 2-4 key URLs inline.
- Avoid repetition or generic filler.
Return only the final readable summary.
"""

    summary = ask_gemini_raw(
        search_summary_prompt,
        key_envs=["GEMINI_SUMMARIZER_KEY", "GEMINI_API_KEY"],
        model_envs=[
            "GEMINI_SUMMARIZER_MODELS",
            "GEMINI_MODEL_PREFERENCE",
            "GEMINI_MODELS",
        ],
    )

    if summary.startswith("[LLM call failed"):
        summary = (
            "LLM summarization temporarily unavailable; showing raw snippets:\n\n"
            + joined_snippets
        )

    sources = ""
    if source_links:
        sources = "\n\nSources:\n" + "\n".join(f"- {url}" for url in source_links[:5])

    final_summary = summary.strip() or "[Web search error: empty summary returned]"
    return final_summary + sources


def _tavily_search(query: str, num_results: int):
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_key:
        return [], "TAVILY_API_KEY not set in environment."
    try:
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query=query, max_results=num_results)
        return results.get("results", []), None
    except Exception as e:
        return [], f"Tavily error: {e}"


def _duckduckgo_search(query: str, num_results: int):
    if DDGS is None:
        return [], "duckduckgo-search package not available."
    try:
        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(
                    query,
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=num_results,
                )
            )
    except Exception as e:
        return [], f"DuckDuckGo error: {e}"
    normalized = []
    for r in raw_results:
        normalized.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "content": r.get("body", r.get("snippet", "")),
            }
        )
    return normalized, None


def web_search(query: str, context: str = "", num_results: int = 5) -> str:
    """
    Performs a context-aware web search with Tavily first, then DuckDuckGo fallback.
    """
    errors = []
    snippets = []
    source_links = []

    tavily_results, err = _tavily_search(query, num_results)
    if err:
        errors.append(err)

    if tavily_results:
        usable = tavily_results[:num_results]
    else:
        ddg_results, err = _duckduckgo_search(query, num_results)
        if err:
            errors.append(err)
        usable = ddg_results[:num_results]

    for r in usable:
        url = r.get("url", "")
        title = r.get("title", "")
        content = (r.get("content", "") or "")[:800]
        if url:
            source_links.append(url)
        snippets.append(f"Title: {title}\nURL: {url}\nContent:\n{content}\n---")

    if not snippets:
        if errors:
            return "[Web search error: " + " | ".join(errors) + "]"
        return f"No results found for '{query}'."

    return _summarize_search_results(query, context, snippets, source_links)
