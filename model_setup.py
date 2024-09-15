class CustomModel(nn.Module):
    def __init__(self, bert_model):
        super(CustomModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits