import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
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


# Load datasets
dataset_1 = load_dataset(
    "hkust-nlp/deita-quality-scorer-data", split='validation')
dataset_2 = load_dataset(
    "turkish-nlp-suite/vitamins-supplements-reviews", split='train')
dataset_3 = load_dataset(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", split='train')

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

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
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])

# Use DataLoader for optimized data loading
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training arguments with mixed precision
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Set epochs
    per_device_train_batch_size=16,  # Adjust batch size
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,  # Increase logging steps to reduce logging frequency
    evaluation_strategy='epoch',  # Set evaluation strategy to 'epoch'
    save_strategy='epoch',        # Set save strategy to 'epoch'
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=5e-5,  # Increase learning rate
    report_to='none',  # Disable reporting to avoid unnecessary overhead
    fp16=True,  # Enable mixed precision training
    # Use gradient accumulation to simulate larger batch size
    gradient_accumulation_steps=2
)

# Metrics function


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(p.label_ids, preds)
    recall = recall_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds)
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
precision = precision_score(labels[train_size:], preds)
recall = recall_score(labels[train_size:], preds)
f1 = f1_score(labels[train_size:], preds)
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
