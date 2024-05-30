import numpy as np
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, random_split
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

sample_size = 50


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


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


def train_and_evaluate(dataset_name, dataset_split, sample_size, language, title):
    dataset = load_and_sample_dataset(dataset_name, dataset_split, sample_size)
    processed_data = [preprocess_text(entry['input']) if 'input' in entry else preprocess_text(
        entry['text'], language=language) for entry in dataset]
    labels = np.array([i % 2 for i in range(len(processed_data))])

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(processed_data, truncation=True,
                          padding=True, max_length=64)

    dataset = TextDataset(encodings, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, num_workers=4, pin_memory=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased')

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=1e-4,
        report_to='none',
        fp16=True,
        gradient_accumulation_steps=1
    )

    class LogTrainingLossCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.train_losses = []
            self.eval_losses = []
            self.eval_accuracies = []

        def on_epoch_end(self, args, state, control, **kwargs):
            if state.log_history:
                last_log = state.log_history[-1]
                if 'loss' in last_log:
                    self.train_losses.append(last_log['loss'])
                if 'eval_loss' in last_log:
                    self.eval_losses.append(last_log['eval_loss'])
                if 'eval_accuracy' in last_log:
                    self.eval_accuracies.append(last_log['eval_accuracy'])

    log_callback = LogTrainingLossCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[log_callback]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer.train()
    eval_result = trainer.evaluate()

    print(f"Evaluation results for {title}:", eval_result)

    plot_training_history(log_callback, title)

    return log_callback.train_losses, log_callback.eval_losses, log_callback.eval_accuracies


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(
        p.label_ids, preds, average='weighted', zero_division=1)
    recall = recall_score(p.label_ids, preds, average='weighted')
    f1 = f1_score(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def plot_training_history(log_callback, title):
    epochs = range(1, len(log_callback.train_losses) + 1)
    train_losses = log_callback.train_losses
    eval_losses = log_callback.eval_losses
    eval_accuracies = log_callback.eval_accuracies

    min_length = min(len(train_losses), len(eval_losses), len(eval_accuracies))

    epochs = list(epochs)[:min_length]
    train_losses = train_losses[:min_length]
    eval_losses = eval_losses[:min_length]
    eval_accuracies = eval_accuracies[:min_length]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, eval_losses, label='Validation Loss')
    plt.title(f'Loss - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


train_losses_1, eval_losses_1, eval_accuracies_1 = train_and_evaluate(
    "hkust-nlp/deita-quality-scorer-data", 'validation', sample_size, 'english', 'Dataset 1')

train_losses_2, eval_losses_2, eval_accuracies_2 = train_and_evaluate(
    "turkish-nlp-suite/vitamins-supplements-reviews", 'train', sample_size, 'turkish', 'Dataset 2')

train_losses_3, eval_losses_3, eval_accuracies_3 = train_and_evaluate(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", 'train', sample_size, 'turkish', 'Dataset 3')


def plot_overall_training_history(train_losses_list, eval_losses_list, eval_accuracies_list):
    epochs = range(1, len(train_losses_list[0]) + 1)
    avg_train_losses = np.mean(train_losses_list, axis=0)
    avg_eval_losses = np.mean(eval_losses_list, axis=0)
    avg_eval_accuracies = np.mean(eval_accuracies_list, axis=0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_train_losses, label='Training Loss')
    plt.plot(epochs, avg_eval_losses, label='Validation Loss')
    plt.title('Overall Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_eval_accuracies, label='Validation Accuracy')
    plt.title('Overall Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_overall_training_history([train_losses_1, train_losses_2, train_losses_3],
                              [eval_losses_1, eval_losses_2, eval_losses_3],
                              [eval_accuracies_1, eval_accuracies_2, eval_accuracies_3])
