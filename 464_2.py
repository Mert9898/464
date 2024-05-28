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

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text, language='english'):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(language))
    tokens = [lemmatizer.lemmatize(stemmer.stem(token.lower(
    ))) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Load and sample datasets


def load_and_sample_dataset(dataset_name, split, sample_size):
    dataset = load_dataset(dataset_name, split=split)
    actual_sample_size = min(len(dataset), sample_size)
    indices = np.random.choice(len(dataset), actual_sample_size, replace=False)
    dataset = dataset.select(indices)
    return dataset


# Adjust sample sizes dynamically based on dataset availability
sample_size = 500  # Further reduced sample size

dataset_1 = load_and_sample_dataset(
    "hkust-nlp/deita-quality-scorer-data", 'validation', sample_size)
dataset_2 = load_and_sample_dataset(
    "turkish-nlp-suite/vitamins-supplements-reviews", 'train', sample_size)
dataset_3 = load_and_sample_dataset(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", 'train', sample_size)

# Preprocess datasets
processed_data_1 = [preprocess_text(entry['input']) for entry in dataset_1]
processed_data_2 = [preprocess_text(
    entry['product_name'], language='turkish') for entry in dataset_2]
processed_data_3 = [preprocess_text(
    entry['movie'], language='turkish') for entry in dataset_3]

# Concatenate datasets for tokenization
texts = processed_data_1 + processed_data_2 + processed_data_3
labels_1 = np.array([i % 2 for i in range(len(processed_data_1))])
labels_2 = np.array([i % 2 for i in range(len(processed_data_2))])
labels_3 = np.array([i % 2 for i in range(len(processed_data_3))])
labels = np.concatenate([labels_1, labels_2, labels_3])

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize data with a reduced max length
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

# Convert to dataset


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

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_indices, val_indices = torch.utils.data.random_split(
    range(len(dataset)), [train_size, val_size])
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Use DataLoader for optimized data loading
train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=16, num_workers=8, pin_memory=True)

# Initialize model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')

# Training arguments with increased learning rate and optimized settings
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduce number of epochs
    per_device_train_batch_size=16,  # Adjust batch size
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,  # Increase logging steps to reduce logging frequency
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=5e-4,  # Increase learning rate significantly
    report_to='none',  # Disable reporting to avoid unnecessary overhead
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=4  # Adjust gradient accumulation steps
)

# Metrics function


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    f1 = f1_score(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Ensure GPU is used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

# Make predictions on the validation set
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# Calculate metrics
precision = precision_score(labels[train_size:], preds, average='weighted')
recall = recall_score(labels[train_size:], preds, average='weighted')
f1 = f1_score(labels[train_size:], preds, average='weighted')
accuracy = accuracy_score(labels[train_size:], preds)

print(f"Precision: {precision:.4f}, Recall: {
      recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Example entries
example_entry_1 = dataset_1[0]
example_entry_2 = dataset_2[0]
example_entry_3 = dataset_3[0]

print("Example Entry from Dataset 1:", example_entry_1)
print("Example Entry from Dataset 2:", example_entry_2)
print("Example Entry from Dataset 3:", example_entry_3)
