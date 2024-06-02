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
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


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


sample_size = 1000

dataset_1 = load_and_sample_dataset(
    "hkust-nlp/deita-quality-scorer-data", 'validation', sample_size)
dataset_2 = load_and_sample_dataset(
    "turkish-nlp-suite/vitamins-supplements-reviews", 'train', sample_size)
dataset_3 = load_and_sample_dataset(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", 'train', sample_size)

processed_data_1 = [preprocess_text(entry['input']) for entry in dataset_1]
processed_data_2 = [preprocess_text(
    entry['text'], language='turkish') for entry in dataset_2]
processed_data_3 = [preprocess_text(
    entry['text'], language='turkish') for entry in dataset_3]

texts_1 = processed_data_1
texts_2 = processed_data_2
texts_3 = processed_data_3

labels_1 = np.array([i % 2 for i in range(len(processed_data_1))])
labels_2 = np.array([i % 2 for i in range(len(processed_data_2))])
labels_3 = np.array([i % 2 for i in range(len(processed_data_3))])

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

encodings_1 = tokenizer(texts_1, truncation=True, padding=True, max_length=64)
encodings_2 = tokenizer(texts_2, truncation=True, padding=True, max_length=64)
encodings_3 = tokenizer(texts_3, truncation=True, padding=True, max_length=64)


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


def plot_training_history(train_losses, eval_losses, eval_accuracies, eval_f1s, eval_precisions, eval_recalls, title):
    epochs = range(1, len(train_losses) + 1)
    min_length = min(len(train_losses), len(eval_losses), len(
        eval_accuracies), len(eval_f1s), len(eval_precisions), len(eval_recalls))

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs[:min_length], train_losses[:min_length],
             label='Training Loss')
    plt.plot(epochs[:min_length], eval_losses[:min_length],
             label='Validation Loss')
    plt.title(f'Loss - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs[:min_length], eval_accuracies[:min_length],
             label='Validation Accuracy')
    plt.title(f'Accuracy - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs[:min_length], eval_f1s[:min_length], label='Validation F1')
    plt.plot(epochs[:min_length], eval_precisions[:min_length],
             label='Validation Precision')
    plt.plot(epochs[:min_length], eval_recalls[:min_length],
             label='Validation Recall')
    plt.title(f'F1, Precision, Recall - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.show()


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    f1 = f1_score(p.label_ids, preds, average='weighted')
    accuracy = accuracy_score(p.label_ids, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }


def train_and_evaluate(encodings, labels, title):
    dataset = TextDataset(encodings, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased')

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=5e-5,
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
            self.eval_f1s = []
            self.eval_precisions = []
            self.eval_recalls = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if 'loss' in logs:
                    self.train_losses.append(logs['loss'])
                if 'eval_loss' in logs:
                    self.eval_losses.append(logs['eval_loss'])
                if 'eval_accuracy' in logs:
                    self.eval_accuracies.append(logs['eval_accuracy'])
                if 'eval_f1' in logs:
                    self.eval_f1s.append(logs['eval_f1'])
                if 'eval_precision' in logs:
                    self.eval_precisions.append(logs['eval_precision'])
                if 'eval_recall' in logs:
                    self.eval_recalls.append(logs['eval_recall'])

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

    model.save_pretrained(f'./results/trained_model_{title.replace(" ", "_")}')
    tokenizer.save_pretrained(
        f'./results/trained_model_{title.replace(" ", "_")}')

    np.savez(f'./results/training_metrics_{title.replace(" ", "_")}.npz',
             train_losses=log_callback.train_losses,
             eval_losses=log_callback.eval_losses,
             eval_accuracies=log_callback.eval_accuracies,
             eval_f1s=log_callback.eval_f1s,
             eval_precisions=log_callback.eval_precisions,
             eval_recalls=log_callback.eval_recalls)

    eval_result = trainer.evaluate()
    print(f"Evaluation results for {title}: {eval_result}")

    plot_training_history(log_callback.train_losses,
                          log_callback.eval_losses, log_callback.eval_accuracies,
                          log_callback.eval_f1s, log_callback.eval_precisions, log_callback.eval_recalls, title)

    return (log_callback.train_losses, log_callback.eval_losses, log_callback.eval_accuracies,
            log_callback.eval_f1s, log_callback.eval_precisions, log_callback.eval_recalls)


def plot_overall_training_history(all_train_losses, all_eval_losses, all_eval_accuracies, all_eval_f1s, all_eval_precisions, all_eval_recalls):
    min_length = min(min(len(l) for l in all_train_losses), min(len(l) for l in all_eval_losses), min(len(l) for l in all_eval_accuracies),
                     min(len(l) for l in all_eval_f1s), min(len(l) for l in all_eval_precisions), min(len(l) for l in all_eval_recalls))
    epochs = range(1, min_length + 1)
    avg_train_losses = np.mean([losses[:min_length]
                               for losses in all_train_losses], axis=0)
    avg_eval_losses = np.mean([losses[:min_length]
                              for losses in all_eval_losses], axis=0)
    avg_eval_accuracies = np.mean([accs[:min_length]
                                  for accs in all_eval_accuracies], axis=0)
    avg_eval_f1s = np.mean([f1s[:min_length] for f1s in all_eval_f1s], axis=0)
    avg_eval_precisions = np.mean(
        [precisions[:min_length] for precisions in all_eval_precisions], axis=0)
    avg_eval_recalls = np.mean([recalls[:min_length]
                               for recalls in all_eval_recalls], axis=0)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, avg_train_losses, label='Training Loss')
    plt.plot(epochs, avg_eval_losses, label='Validation Loss')
    plt.title('Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, avg_eval_accuracies, label='Validation Accuracy')
    plt.title('Average Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, avg_eval_f1s, label='Validation F1')
    plt.plot(epochs, avg_eval_precisions, label='Validation Precision')
    plt.plot(epochs, avg_eval_recalls, label='Validation Recall')
    plt.title('Average F1, Precision, Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.show()


def main():
    all_train_losses, all_eval_losses, all_eval_accuracies, all_eval_f1s, all_eval_precisions, all_eval_recalls = [], [], [], [], [], []
    for i, (encodings, labels, title) in enumerate([(encodings_1, labels_1, 'Dataset 1'),
                                                    (encodings_2, labels_2,
                                                     'Dataset 2'),
                                                    (encodings_3, labels_3, 'Dataset 3')]):
        model_dir = f'./results/trained_model_{title.replace(" ", "_")}'
        metrics_file = f'./results/training_metrics_{
            title.replace(" ", "_")}.npz'
        if not os.path.exists(model_dir):
            print(f"Training the model for {title} as it doesn't exist.")
            train_losses, eval_losses, eval_accuracies, eval_f1s, eval_precisions, eval_recalls = train_and_evaluate(
                encodings, labels, title)
        else:
            if not os.path.exists(metrics_file):
                print(f"Metrics file not found for {
                      title}. Retraining the model.")
                train_losses, eval_losses, eval_accuracies, eval_f1s, eval_precisions, eval_recalls = train_and_evaluate(
                    encodings, labels, title)
            else:
                print(
                    f"Loading the previously trained model and metrics for {title}.")
                metrics = np.load(metrics_file)
                train_losses = metrics['train_losses'].tolist()
                eval_losses = metrics['eval_losses'].tolist()
                eval_accuracies = metrics['eval_accuracies'].tolist()
                eval_f1s = metrics['eval_f1s'].tolist()
                eval_precisions = metrics['eval_precisions'].tolist()
                eval_recalls = metrics['eval_recalls'].tolist()

        all_train_losses.append(train_losses)
        all_eval_losses.append(eval_losses)
        all_eval_accuracies.append(eval_accuracies)
        all_eval_f1s.append(eval_f1s)
        all_eval_precisions.append(eval_precisions)
        all_eval_recalls.append(eval_recalls)

        plot_training_history(train_losses, eval_losses, eval_accuracies,
                              eval_f1s, eval_precisions, eval_recalls, title)

    plot_overall_training_history(all_train_losses, all_eval_losses,
                                  all_eval_accuracies, all_eval_f1s, all_eval_precisions, all_eval_recalls)


if __name__ == "__main__":
    main()


def load_model_and_infer(model_path, tokenizer, new_texts, language='english'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_texts = [preprocess_text(
        text, language=language) for text in new_texts]
    encodings = tokenizer(processed_texts, truncation=True,
                          padding=True, max_length=64)

    class NewTextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()}
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])

    new_dataset = NewTextDataset(encodings)

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    data_loader = DataLoader(new_dataset, batch_size=8)

    all_preds = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds


new_texts = ["This is a new text to classify.",
             "Another text for classification."]
predictions = load_model_and_infer(
    './results/trained_model_Dataset_1', tokenizer, new_texts, language='english')
print(f"Predictions: {predictions}")
