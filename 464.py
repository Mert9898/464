import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

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
    return tokens


def load_and_sample_dataset(dataset_name, split, sample_size):
    dataset = load_dataset(dataset_name, split=split)
    actual_sample_size = min(len(dataset), sample_size)
    indices = np.random.choice(len(dataset), actual_sample_size, replace=False)
    dataset = dataset.select(indices)
    return dataset


sample_size = 200

dataset_1 = load_and_sample_dataset(
    "hkust-nlp/deita-quality-scorer-data", 'validation', sample_size)
dataset_2 = load_and_sample_dataset(
    "turkish-nlp-suite/vitamins-supplements-reviews", 'train', sample_size)
dataset_3 = load_and_sample_dataset(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", 'train', sample_size)

processed_data_1 = [' '.join(preprocess_text(entry['input']))
                    for entry in dataset_1]
processed_data_2 = [' '.join(preprocess_text(
    entry['product_name'], language='turkish')) for entry in dataset_2]
processed_data_3 = [' '.join(preprocess_text(
    entry['movie'], language='turkish')) for entry in dataset_3]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(processed_data_1 + processed_data_2 + processed_data_3)

sequences_1 = tokenizer.texts_to_sequences(processed_data_1)
sequences_2 = tokenizer.texts_to_sequences(processed_data_2)
sequences_3 = tokenizer.texts_to_sequences(processed_data_3)

max_seq_length = 100
data_1 = pad_sequences(sequences_1, maxlen=max_seq_length)
data_2 = pad_sequences(sequences_2, maxlen=max_seq_length)
data_3 = pad_sequences(sequences_3, maxlen=max_seq_length)


def create_model(input_length):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=32, input_length=input_length),
        LSTM(64, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model_1 = create_model(max_seq_length)
model_2 = create_model(max_seq_length)
model_3 = create_model(max_seq_length)

labels_1 = np.array([i % 2 for i in range(len(data_1))])
labels_2 = np.array([i % 2 for i in range(len(data_2))])
labels_3 = np.array([i % 2 for i in range(len(data_3))])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=1, restore_best_weights=True)

history_1 = model_1.fit(data_1, labels_1, epochs=3, batch_size=128,
                        validation_split=0.2, callbacks=[early_stopping])
history_2 = model_2.fit(data_2, labels_2, epochs=3, batch_size=128,
                        validation_split=0.2, callbacks=[early_stopping])
history_3 = model_3.fit(data_3, labels_3, epochs=3, batch_size=128,
                        validation_split=0.2, callbacks=[early_stopping])


def evaluate_model(model, data, labels):
    predictions = (model.predict(data) > 0.5).astype("int32")
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    return precision, recall, f1, accuracy


precision_1, recall_1, f1_1, accuracy_1 = evaluate_model(
    model_1, data_1, labels_1)
precision_2, recall_2, f1_2, accuracy_2 = evaluate_model(
    model_2, data_2, labels_2)
precision_3, recall_3, f1_3, accuracy_3 = evaluate_model(
    model_3, data_3, labels_3)

print("Model 1 - Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Accuracy: {:.4f}".format(
    precision_1, recall_1, f1_1, accuracy_1))
print("Model 2 - Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Accuracy: {:.4f}".format(
    precision_2, recall_2, f1_2, accuracy_2))
print("Model 3 - Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Accuracy: {:.4f}".format(
    precision_3, recall_3, f1_3, accuracy_3))


def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss ' + title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy ' + title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_training_history(history_1, 'Model 1')
plot_training_history(history_2, 'Model 2')
plot_training_history(history_3, 'Model 3')

example_entry_1 = dataset_1[0]
example_entry_2 = dataset_2[0]
example_entry_3 = dataset_3[0]

print("Example Entry from Dataset 1:", example_entry_1)
print("Example Entry from Dataset 2:", example_entry_2)
print("Example Entry from Dataset 3:", example_entry_3)
