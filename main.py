# main.py

from dag_builder import build_graph
import logging
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/classification.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

graph = build_graph().compile()

def run_cli():
    print("=== Sentiment Classifier with Self-Healing Fallback ===\n")
    while True:
        text = input("Enter a review (or type 'exit'): ").strip()
        if text.lower() == "exit":
            print("bye!")
            break

        result = graph.invoke({"input_text": text})


        label = result.get("final_label") or result.get("prediction")
        confidence = result["confidence"]
        corrected = result.get("corrected", False)

        print("\n[Final Result]")
        print(f"Label: {label}")
        print(f"Confidence: {confidence:.2f}")
        if corrected:
            print("Note: Label was corrected via fallback.")
        print("-" * 50)

if __name__ == "__main__":
    run_cli()
