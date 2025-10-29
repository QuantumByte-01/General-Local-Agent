from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    plan_node,
    execute_node,
    verify_node,
    route_after_verify_node,
    repair_subtask_node,
    finalize_node,
    route_router,
)


def build_agent_graph() -> StateGraph:
    
    builder = StateGraph(AgentState)

    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("verify", verify_node)
    builder.add_node("route_after_verify", route_after_verify_node)
    builder.add_node("repair_subtask", repair_subtask_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("plan")

    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "verify")
    builder.add_edge("verify", "route_after_verify")

    builder.add_conditional_edges(
        "route_after_verify",
        route_router,
        {
            "execute": "execute",
            "repair_subtask": "repair_subtask",
            "finalize": "finalize",
        },
    )

    
    builder.add_edge("repair_subtask", "execute")

    builder.add_edge("finalize", END)

    return builder.compile()


def build_demo_graph() -> StateGraph:
  
    builder = StateGraph(AgentState)

    builder.add_node("plan", lambda s: s)
    builder.add_node("execute", lambda s: s)
    builder.add_node("verify", lambda s: s)
    builder.add_node("route_after_verify", lambda s: s)
    builder.add_node("repair_subtask", lambda s: s)
    builder.add_node("finalize", lambda s: s)

    builder.set_entry_point("plan")

    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "verify")
    builder.add_edge("verify", "route_after_verify")

    builder.add_conditional_edges(
        "route_after_verify",
        lambda st: st.get("route", "finalize"),
        {
            "execute": "execute",
            "repair_subtask": "repair_subtask",
            "finalize": "finalize",
        },
    )

    builder.add_edge("repair_subtask", "execute")
    builder.add_edge("finalize", END)

    return builder.compile()
