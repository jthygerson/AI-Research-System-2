training_args = TrainingArguments(
    per_device_train_batch_size=32,  # Increased batch size
    per_device_eval_batch_size=32,   # Increased batch size
    learning_rate=3e-5,              # Reduced learning rate
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)