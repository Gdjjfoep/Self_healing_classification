from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./fine_tuned_model"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}

def main():
    # Load dataset
    dataset = load_dataset("imdb")
    # For faster training, use a small subset
    small_train = dataset["train"].shuffle(seed=42).select(range(200))
    small_test = dataset["test"].shuffle(seed=42).select(range(100))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_train = small_train.map(tokenize_function, batched=True)
    tokenized_test = small_test.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

