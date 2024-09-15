from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # increased number of training epochs
    per_device_train_batch_size=32,  # increased batch size for training
    per_device_eval_batch_size=32,   # increased batch size for evaluation
    learning_rate=3e-5,              # decreased learning rate for finer updates
    logging_dir='./logs',            # directory for storing logs
    weight_decay=0.01,               # added weight decay for regularization
    logging_steps=10,                # more frequent logging
    save_steps=500,                  # save checkpoint every 500 steps
    evaluation_strategy="steps",     # evaluate during training at each save step
    save_total_limit=5,              # limit the total amount of checkpoints
    load_best_model_at_end=True,     # load the best model at the end of training
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,           # evaluation dataset
    data_collator=data_collator,         # added data collator for dynamic padding
)

# Adding Data Augmentation (assuming this is a text classification task)
from nlpaug.augmenter.word import SynonymAug

def augment_data(dataset):
    aug = SynonymAug(aug_src='wordnet', model_path='wordnet')
    augmented_texts = [aug.augment(text) for text in dataset["texts"]]
    return augmented_texts

# Apply augmentation to the training dataset
train_dataset["texts"] = augment_data(train_dataset)

trainer.train()