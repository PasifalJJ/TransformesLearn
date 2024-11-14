if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("yelp_review_full")

    print(dataset["train"][100])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    from transformers import AutoModelForSequenceClassification
    # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5).to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

    import numpy as np
    import evaluate

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",use_cpu=True,
                                      per_gpu_eval_batch_size=1, per_device_train_batch_size=1, per_gpu_train_batch_size=1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()