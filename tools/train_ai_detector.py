import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import pandas as pd
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
import torch
from modules.text_preprocessor import clean_text

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    ai_dir = BASE_DIR / "datasets/ai_assist_dataset"
    human_dir = BASE_DIR / "datasets/benchmark_dataset"
    
    data = []
    for path in ai_dir.rglob("*.txt"):
        text = clean_text(path.read_text(errors="ignore"))
        data.append({"text": text, "label": 1})
    for path in human_dir.rglob("*.txt"):
        text = clean_text(path.read_text(errors="ignore"))
        data.append({"text": text, "label": 0})
    
    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
    print(f"Loaded {len(df)} samples: {df['label'].value_counts().to_dict()}")
    
    df = df.dropna(subset=['text', 'label'])
    
    train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    eval_dataset = Dataset.from_pandas(eval_df).map(tokenize, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", num_labels=2)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels, average='binary')
    
    args = TrainingArguments(
        output_dir="./models/fine_tuned_ai_detector",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs/ai_detector',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model()
    print("Fine-tuned AI detector saved to models/fine_tuned_ai_detector")
    results = trainer.evaluate()
    print(f"Final eval F1: {results['eval_f1']}")

if __name__ == "__main__":
    main()

