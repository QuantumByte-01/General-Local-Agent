import os
import uuid
import gradio as gr
from dotenv import load_dotenv

from agent.runtime import run_agent_once
from agent.graph import build_demo_graph


load_dotenv()  


def agent_turn(user_msg: str, history: list, session_state: dict):
 
    if not session_state or session_state.get("session_id") is None:
        session_state = {"session_id": str(uuid.uuid4())}

    convo_history = history + [{"role": "user", "content": user_msg}]

    base_dir = os.getenv(
        "AGENT_BASE_DIR"
    )

    final_answer, _debug_info = run_agent_once(
        goal=user_msg,
        chat_history=convo_history,
        base_dir=base_dir,
        max_retries=3,
    )

    assistant_msg = final_answer
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

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
