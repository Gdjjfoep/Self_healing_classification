📌 Self-Healing Sentiment Classifier
This project implements a LangGraph-based sentiment classifier with a self-healing fallback mechanism. It uses a fine-tuned DistilBERT model to classify movie reviews and asks for user input when prediction confidence is low.

🛠️ Features
Fine-tuned transformer model (DistilBERT)

LangGraph DAG with:

InferenceNode

ConfidenceCheckNode

FallbackNode

CLI for user interaction and fallback correction

Structured logging of all steps

⚠️ Hardware Constraints
Due to limited hardware (Intel i3 7th Gen laptop), we:

Used a smaller subset of IMDb dataset (~1000 samples)

Trained for 1 epoch only

Skipped validation to save time

As a result, model confidence may default to ~0.5 often.

