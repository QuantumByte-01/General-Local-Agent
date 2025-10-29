import os
import uuid
import gradio as gr
from dotenv import load_dotenv

from agent.runtime import run_agent_once, build_demo_graph

load_dotenv() 

def agent_turn(user_msg: str, history: list, session_state: dict):

    if not session_state or session_state.get("session_id") is None:
        session_state = {
            "session_id": str(uuid.uuid4()),
        }

    # Conversation context for planner
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

    assistant_text = final_answer

    new_history = convo_history + [
        {"role": "assistant", "content": assistant_text}
    ]

    return new_history, "", session_state


def main():
    demo_graph = build_demo_graph()
    print("AGENT FLOW GRAPH (LangGraph ASCII) ")
    print(demo_graph.get_graph().draw_ascii())

    with gr.Blocks() as demo:
        gr.Markdown("# 🖥️ Local OS Agent")

        chatbot = gr.Chatbot(
            label="Local Agent",
            height=400,
            type="messages"  
        )

        user_in = gr.Textbox(
            label="Your request",
            placeholder=(
                "Examples:\n"
                "- search heart.csv and print the first 5 rows\n"
                "- list all pdf files in the base dir\n"
                "- how much free space is on C:?\n"
                "- count how many folders are in Downloads\n"
            ),
            lines=2,
        )

        state = gr.State({})  

        send_btn = gr.Button("Run")
        send_btn.click(
            fn=agent_turn,
            inputs=[user_in, chatbot, state],
            outputs=[chatbot, user_in, state],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
