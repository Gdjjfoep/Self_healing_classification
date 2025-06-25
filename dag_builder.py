from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph
from nodes import inference_node, confidence_check_node, fallback_node



class State(TypedDict):
    input_text: str
    prediction: str
    confidence: float
    user_clarification: str
    final_label: str

def build_graph():
    builder = StateGraph(State)

    builder.add_node("inference", inference_node)
    builder.add_node("confidence_check", confidence_check_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("END", lambda x: x)


    builder.set_entry_point("inference")
    builder.add_edge("inference", "confidence_check")
    builder.add_conditional_edges(
    "confidence_check",
    lambda state: "fallback" if state["confidence"] < 0.7 else "END"
)

    builder.add_edge("fallback", "END")

    return builder
