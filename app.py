import os
import uuid
import gradio as gr
from dotenv import load_dotenv

from agent.runtime import run_agent_once, generate_plan_preview
from agent.graph import build_demo_graph


load_dotenv()  


def _normalize_confirmation(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip().lower()
    yes_tokens = {
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "proceed",
        "continue",
        "confirm",
        "go ahead",
        "run",
        "execute",
    }
    no_tokens = {
        "no",
        "n",
        "stop",
        "cancel",
        "abort",
        "wait",
    }
    for token in yes_tokens:
        if normalized == token or normalized.startswith(f"{token} "):
            return "yes"
    if normalized.startswith("yes"):
        return "yes"
    for token in no_tokens:
        if normalized == token or normalized.startswith(f"{token} "):
            return "no"
    if normalized.startswith("no"):
        return "no"
    return ""


def agent_turn(user_msg: str, history: list, session_state: dict):
 
    if not session_state or session_state.get("session_id") is None:
        session_state = {"session_id": str(uuid.uuid4())}

    convo_history = history + [{"role": "user", "content": user_msg}]

    base_dir = os.getenv(
        "AGENT_BASE_DIR"
    )

    pending_plan = session_state.get("pending_plan")
    pending_goal = session_state.get("pending_goal")

    if pending_plan:
        decision = _normalize_confirmation(user_msg)
        if decision == "yes":
            final_answer, _debug_info = run_agent_once(
                goal=pending_goal or session_state.get("last_goal") or "",
                chat_history=convo_history,
                base_dir=base_dir,
                max_retries=3,
                existing_plan=pending_plan,
            )
            assistant_msg = final_answer
            session_state.pop("pending_plan", None)
            session_state.pop("pending_goal", None)
        elif decision == "no":
            assistant_msg = (
                "Okay, I cancelled that plan. Share new instructions when ready."
            )
            session_state.pop("pending_plan", None)
            session_state.pop("pending_goal", None)
        else:
            assistant_msg = (
                "Please reply with yes to run the proposed plan or no to revise it."
            )
        new_history = convo_history + [{"role": "assistant", "content": assistant_msg}]
        return new_history, "", session_state

    plan, plan_summary = generate_plan_preview(
        goal=user_msg,
        chat_history=convo_history,
        base_dir=base_dir,
    )

    session_state["pending_plan"] = plan
    session_state["pending_goal"] = user_msg
    session_state["last_goal"] = user_msg

    assistant_msg = plan_summary
    new_history = convo_history + [{"role": "assistant", "content": assistant_msg}]

    return new_history, "", session_state


def main():
    demo_graph = build_demo_graph()
    print("=== AGENT FLOW GRAPH (LangGraph ASCII) ===")
    print(demo_graph.get_graph().draw_ascii())
    print("=========================================\n")

    with gr.Blocks() as demo:
        gr.Markdown("A General Local Agent for Computer Task Execution")

        chatbot = gr.Chatbot(
            label="Local Agent",
            height=400,
            type="messages",  
        )

        user_in = gr.Textbox(
            label="Your request",
            placeholder=(""
            ),
            lines=2,
        )

        session_state = gr.State({})

        run_btn = gr.Button("Run")

        run_btn.click(
            fn=agent_turn,
            inputs=[user_in, chatbot, session_state],
            outputs=[chatbot, user_in, session_state],
        )

    demo.launch(server_name="0.0.0.0", server_port=7869, debug=True)


if __name__ == "__main__":
    main()
