import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset
import os

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


sample_size = 10000

datasets = {
    "Dataset 1": load_and_sample_dataset("hkust-nlp/deita-quality-scorer-data", 'validation', sample_size),
    "Dataset 2": load_and_sample_dataset("turkish-nlp-suite/vitamins-supplements-reviews", 'train', sample_size),
    "Dataset 3": load_and_sample_dataset("turkish-nlp-suite/beyazperde-top-300-movie-reviews", 'train', sample_size)
}

all_texts = []
all_labels = []

histories = {}

for dataset_name, dataset in datasets.items():
    print(f"{dataset_name} first entry:", dataset[0])

    if dataset_name == "Dataset 1":
        processed_data = [preprocess_text(entry['input']) for entry in dataset]
    elif dataset_name == "Dataset 2":
        processed_data = [preprocess_text(
            entry['text'], language='turkish') for entry in dataset]
    elif dataset_name == "Dataset 3":
        processed_data = [preprocess_text(
            entry['text'], language='turkish') for entry in dataset]

    labels = np.array([i % 2 for i in range(len(processed_data))])

    all_texts.extend(processed_data)
    all_labels.extend(labels)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    tokenizer = Tokenizer(num_words=10000, lower=True, oov_token='UNK')
    tokenizer.fit_on_texts(processed_data)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    sequences = tokenizer.texts_to_sequences(processed_data)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42)

    model_path = 'best_model_' + dataset_name + '.keras'
    history_path = 'history_' + dataset_name + '.npz'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f'Loaded pre-trained model for {dataset_name}')
        if os.path.exists(history_path):
            history_data = np.load(history_path)
            history = {
                'loss': history_data['loss'].tolist(),
                'val_loss': history_data['val_loss'].tolist(),
                'accuracy': history_data['accuracy'].tolist(),
                'val_accuracy': history_data['val_accuracy'].tolist(),
                'precision': history_data['precision'].tolist(),
                'val_precision': history_data['val_precision'].tolist(),
                'recall': history_data['recall'].tolist(),
                'val_recall': history_data['val_recall'].tolist(),
                'f1_score': history_data['f1_score'].tolist(),
                'val_f1_score': history_data['val_f1_score'].tolist(),
            }
            histories[dataset_name] = history
    else:
        model = Sequential()
        model.add(Embedding(vocab_size, 100))
        model.add(SpatialDropout1D(0.5))
        model.add(LSTM(128, return_sequences=True,
                  dropout=0.4, recurrent_dropout=0.4))
        model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

        model.compile(loss='binary_crossentropy', optimizer=RMSprop(
            learning_rate=1e-4), metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=0.0001, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            model_path, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3, callbacks=[
                            early_stopping, model_checkpoint, reduce_lr])

        predictions_train = (model.predict(X_train) > 0.5).astype(int)
        predictions_val = (model.predict(X_test) > 0.5).astype(int)

        train_precision = precision_score(y_train, predictions_train)
        val_precision = precision_score(y_test, predictions_val)
        train_recall = recall_score(y_train, predictions_train)
        val_recall = recall_score(y_test, predictions_val)
        train_f1 = f1_score(y_train, predictions_train)
        val_f1 = f1_score(y_test, predictions_val)

        histories[dataset_name] = history.history
        histories[dataset_name].update({
            'precision': [train_precision] * len(history.history['loss']),
            'val_precision': [val_precision] * len(history.history['loss']),
            'recall': [train_recall] * len(history.history['loss']),
            'val_recall': [val_recall] * len(history.history['loss']),
            'f1_score': [train_f1] * len(history.history['loss']),
            'val_f1_score': [val_f1] * len(history.history['loss']),
        })

        np.savez(history_path,
                 loss=history.history['loss'],
                 val_loss=history.history['val_loss'],
                 accuracy=history.history['accuracy'],
                 val_accuracy=history.history['val_accuracy'],
                 precision=histories[dataset_name]['precision'],
                 val_precision=histories[dataset_name]['val_precision'],
                 recall=histories[dataset_name]['recall'],
                 val_recall=histories[dataset_name]['val_recall'],
                 f1_score=histories[dataset_name]['f1_score'],
                 val_f1_score=histories[dataset_name]['val_f1_score'])

        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
        print(f'{dataset_name} - Loss: {loss}, Accuracy: {accuracy}')
        print(f'Saving training history for {dataset_name}')

    try:
        plt.figure(figsize=(18, 6))
        if dataset_name in histories:
            history = histories[dataset_name]
            plt.subplot(1, 3, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'Loss - {dataset_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(1, 3, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy - {dataset_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.plot(history['precision'], label='Training Precision')
            plt.plot(history['val_precision'], label='Validation Precision')
            plt.plot(history['recall'], label='Training Recall')
            plt.plot(history['val_recall'], label='Validation Recall')
            plt.plot(history['f1_score'], label='Training F1 Score')
            plt.plot(history['val_f1_score'], label='Validation F1 Score')
            plt.title(f'Precision, Recall, F1 Score - {dataset_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Metrics')
            plt.legend()
        plt.show()
    except Exception as e:
        print(f'Failed to display training history for {dataset_name}: {e}')

label_encoder = LabelEncoder()
all_labels = label_encoder.fit_transform(all_labels)

tokenizer = Tokenizer(num_words=10000, lower=True, oov_token='UNK')
tokenizer.fit_on_texts(all_texts)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(all_texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, all_labels, test_size=0.2, random_state=42)

model_path_general = 'best_model_general.keras'
history_path_general = 'history_general.npz'
if os.path.exists(model_path_general):
    model = load_model(model_path_general)
    print('Loaded pre-trained general model')
    if os.path.exists(history_path_general):
        history_data = np.load(history_path_general)
        history = {
            'loss': history_data['loss'].tolist(),
            'val_loss': history_data['val_loss'].tolist(),
            'accuracy': history_data['accuracy'].tolist(),
            'val_accuracy': history_data['val_accuracy'].tolist(),
            'precision': history_data['precision'].tolist(),
            'val_precision': history_data['val_precision'].tolist(),
            'recall': history_data['recall'].tolist(),
            'val_recall': history_data['val_recall'].tolist(),
            'f1_score': history_data['f1_score'].tolist(),
            'val_f1_score': history_data['val_f1_score'].tolist(),
        }
        histories["General"] = history
else:
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(128, return_sequences=True,
              dropout=0.4, recurrent_dropout=0.4))
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(
        learning_rate=1e-4), metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, min_delta=0.0001, restore_best_weights=True)
    model_checkpoint_general = ModelCheckpoint(
        model_path_general, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3, callbacks=[
                        early_stopping, model_checkpoint_general, reduce_lr])

    predictions_train = (model.predict(X_train) > 0.5).astype(int)
    predictions_val = (model.predict(X_test) > 0.5).astype(int)

    train_precision = precision_score(y_train, predictions_train)
    val_precision = precision_score(y_test, predictions_val)
    train_recall = recall_score(y_train, predictions_train)
    val_recall = recall_score(y_test, predictions_val)
    train_f1 = f1_score(y_train, predictions_train)
    val_f1 = f1_score(y_test, predictions_val)

    histories["General"] = history.history
    histories["General"].update({
        'precision': [train_precision] * len(history.history['loss']),
        'val_precision': [val_precision] * len(history.history['loss']),
        'recall': [train_recall] * len(history.history['loss']),
        'val_recall': [val_recall] * len(history.history['loss']),
        'f1_score': [train_f1] * len(history.history['loss']),
        'val_f1_score': [val_f1] * len(history.history['loss']),
    })

    np.savez(history_path_general,
             loss=history.history['loss'],
             val_loss=history.history['val_loss'],
             accuracy=history.history['accuracy'],
             val_accuracy=history.history['val_accuracy'],
             precision=histories["General"]['precision'],
             val_precision=histories["General"]['val_precision'],
             recall=histories["General"]['recall'],
             val_recall=histories["General"]['val_recall'],
             f1_score=histories["General"]['f1_score'],
             val_f1_score=histories["General"]['val_f1_score'])

    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'General - Loss: {loss}, Accuracy: {accuracy}')
    print('Saving general training history')

try:
    plt.figure(figsize=(18, 6))
    if "General" in histories:
        history = histories["General"]
        plt.subplot(1, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss - General')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy - General')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(history['precision'], label='Training Precision')
        plt.plot(history['val_precision'], label='Validation Precision')
        plt.plot(history['recall'], label='Training Recall')
        plt.plot(history['val_recall'], label='Validation Recall')
        plt.plot(history['f1_score'], label='Training F1 Score')
        plt.plot(history['val_f1_score'], label='Validation F1 Score')
        plt.title('Precision, Recall, F1 Score - General')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
    plt.show()
except Exception as e:
    print(f'Failed to display general training history: {e}')

if histories:
    min_len = min([len(history['loss']) for history in histories.values()])
    avg_loss = np.mean([history['loss'][:min_len]
                       for history in histories.values()], axis=0)
    avg_val_loss = np.mean([history['val_loss'][:min_len]
                           for history in histories.values()], axis=0)
    avg_accuracy = np.mean([history['accuracy'][:min_len]
                           for history in histories.values()], axis=0)
    avg_val_accuracy = np.mean(
        [history['val_accuracy'][:min_len] for history in histories.values()], axis=0)
    avg_precision = np.mean([history['precision'][:min_len]
                            for history in histories.values()], axis=0)
    avg_val_precision = np.mean([history['val_precision'][:min_len]
                                for history in histories.values()], axis=0)
    avg_recall = np.mean([history['recall'][:min_len]
                         for history in histories.values()], axis=0)
    avg_val_recall = np.mean([history['val_recall'][:min_len]
                             for history in histories.values()], axis=0)
    avg_f1_score = np.mean([history['f1_score'][:min_len]
                           for history in histories.values()], axis=0)
    avg_val_f1_score = np.mean([history['val_f1_score'][:min_len]
                               for history in histories.values()], axis=0)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(avg_loss, label='Avg Training Loss')
    plt.plot(avg_val_loss, label='Avg Validation Loss')
    plt.title('Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(avg_accuracy, label='Avg Training Accuracy')
    plt.plot(avg_val_accuracy, label='Avg Validation Accuracy')
    plt.title('Average Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(avg_precision, label='Avg Training Precision')
    plt.plot(avg_val_precision, label='Avg Validation Precision')
    plt.plot(avg_recall, label='Avg Training Recall')
    plt.plot(avg_val_recall, label='Avg Validation Recall')
    plt.plot(avg_f1_score, label='Avg Training F1 Score')
    plt.plot(avg_val_f1_score, label='Avg Validation F1 Score')
    plt.title('Average Precision, Recall, F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()


def load_model_and_infer_lstm(model_path, tokenizer, new_texts, language='english'):
    def preprocess_text(text, language='english'):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words(language))
        tokens = [lemmatizer.lemmatize(stemmer.stem(token.lower(
        ))) for token in tokens if token.isalpha() and token not in stop_words]
        return ' '.join(tokens)

    processed_texts = [preprocess_text(
        text, language=language) for text in new_texts]
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    model = load_model(model_path)
    predictions = model.predict(padded_sequences)
    return (predictions > 0.5).astype(int)


new_texts = ["This is a new text to classify.",
             "Another text for classification."]
predictions = load_model_and_infer_lstm(
    model_path_general, tokenizer, new_texts, language='english')
print(f"Predictions: {predictions}")
