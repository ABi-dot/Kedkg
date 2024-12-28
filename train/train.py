from datasets import load_dataset, load_metric
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
dataset = load_dataset('json', data_files='../train/datasets_entity_judge.json')
dataset = dataset['train'].train_test_split(test_size=0.2)

print(dataset)
PATH = '../model/distilbert-base-cased'
DEVICE = "cuda:0"
tokenizer = DistilBertTokenizer.from_pretrained(PATH)

# preprocess
def preprocess_function(examples):
    return tokenizer(examples['question'], examples['entity'], truncation=True, padding='max_length')

encoded_dataset = dataset.map(preprocess_function, batched=True)
model = DistilBertForSequenceClassification.from_pretrained(PATH, num_labels=2).to(DEVICE)
accuracy_metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


# train arguments
training_args = TrainingArguments(
    output_dir='./results_entity_judge',
    evaluation_strategy='epoch',  # 在每个epoch结束时评估模型
    learning_rate=2e-5,
    save_strategy='epoch',  # 在每个epoch结束时保存模型
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',  # 日志保存目录
    logging_steps=10,  # 每10步记录一次日志
    save_steps=500,  # 每500步保存一次模型
    save_total_limit=3,  # 仅保留最近的3个检查点
    load_best_model_at_end=True,  # 在训练结束时加载表现最好的模型
    metric_for_best_model="accuracy",  # 用于评估模型表现的指标
    greater_is_better=True,  # 指标越大越好
    fp16=True  # 使用混合精度训练
)


# Initialize
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print(results)

model.save_pretrained('./results_entity_judge/best_model_entity_judge')
tokenizer.save_pretrained('./results_entity_judge/best_model_entity_judge')