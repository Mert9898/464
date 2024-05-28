import numpy as np
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, Subset
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text, language='english'):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(language))
    tokens = [lemmatizer.lemmatize(stemmer.stem(token.lower(
    ))) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)


def load_and_sample_dataset(dataset_name, split, sample_size):
    dataset = load_dataset(dataset_name, split=split)
    actual_sample_size = min(len(dataset), sample_size)
    indices = np.random.choice(len(dataset), actual_sample_size, replace=False)
    dataset = dataset.select(indices)
    return dataset


sample_size = 100

dataset_1 = load_and_sample_dataset(
    "hkust-nlp/deita-quality-scorer-data", 'validation', sample_size)
dataset_2 = load_and_sample_dataset(
    "turkish-nlp-suite/vitamins-supplements-reviews", 'train', sample_size)
dataset_3 = load_and_sample_dataset(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", 'train', sample_size)

processed_data_1 = [preprocess_text(entry['input']) for entry in dataset_1]
processed_data_2 = [preprocess_text(
    entry['product_name'], language='turkish') for entry in dataset_2]
processed_data_3 = [preprocess_text(
    entry['movie'], language='turkish') for entry in dataset_3]

texts = processed_data_1 + processed_data_2 + processed_data_3
labels_1 = np.array([i % 2 for i in range(len(processed_data_1))])
labels_2 = np.array([i % 2 for i in range(len(processed_data_2))])
labels_3 = np.array([i % 2 for i in range(len(processed_data_3))])
labels = np.concatenate([labels_1, labels_2, labels_3])

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

encodings = tokenizer(texts, truncation=True, padding=True,
                      max_length=64)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


dataset = TextDataset(encodings, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_indices, val_indices = torch.utils.data.random_split(
    range(len(dataset)), [train_size, val_size])
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=32,
                            num_workers=4, pin_memory=True)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=5e-4,
    report_to='none',
    fp16=True,
    gradient_accumulation_steps=2
)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    f1 = f1_score(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

trainer.train()

trainer.save_model('./results/trained_model')

eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

loaded_model = DistilBertForSequenceClassification.from_pretrained(
    './results/trained_model')
loaded_model.to(device)

loaded_trainer = Trainer(
    model=loaded_model,
    args=training_args,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)

eval_result_loaded = loaded_trainer.evaluate()
print("Evaluation results from loaded model:", eval_result_loaded)

predictions = loaded_trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

val_labels = np.array([labels[idx] for idx in val_indices.indices])
preds = preds[:len(val_labels)]

precision = precision_score(val_labels, preds, average='weighted')
recall = recall_score(val_labels, preds, average='weighted')
f1 = f1_score(val_labels, preds, average='weighted')
accuracy = accuracy_score(val_labels, preds)

print(f"Precision: {precision:.4f}, Recall: {
      recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

example_entry_1 = dataset_1[0]
example_entry_2 = dataset_2[0]
example_entry_3 = dataset_3[0]

print("Example Entry from Dataset 1:", example_entry_1)
print("Example Entry from Dataset 2:", example_entry_2)
print("Example Entry from Dataset 3:", example_entry_3)
