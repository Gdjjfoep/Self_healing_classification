# nodes.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging

# Load fine-tuned model
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

CONFIDENCE_THRESHOLD = 0.75
logger = logging.getLogger(__name__)

def inference_node(state):
    input_text = state["input_text"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=1)

    label = "Positive" if prediction.item() == 1 else "Negative"
    confidence_score = confidence.item()

    logger.info(f"[InferenceNode] Predicted label: {label} | Confidence: {confidence_score:.2f}")
    return {
        "text": input_text,
        "label": label,
        "confidence": confidence_score
    }

def confidence_check_node(state):
    confidence = state.get("confidence", 0.0)

    # Just return the unchanged state — routing is handled in the DAG logic
    return state


def fallback_node(state):
    print("\n[FallbackNode] Could you clarify your intent?")
    print(f"Original Text: \"{state['input_text']}\"")
    clarification = input("Was this review positive, negative, or neutral? ").strip().lower()

    # Log or override prediction
    return {
        **state,
        "user_clarification": clarification,
        "final_label": clarification
    }
