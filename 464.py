import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
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


dataset_1 = load_dataset(
    "hkust-nlp/deita-quality-scorer-data", split='validation')
dataset_2 = load_dataset(
    "turkish-nlp-suite/vitamins-supplements-reviews", split='train')
dataset_3 = load_dataset(
    "turkish-nlp-suite/beyazperde-top-300-movie-reviews", split='train')

processed_data_1 = [' '.join(preprocess_text(entry['input']))
                    for entry in dataset_1]
processed_data_2 = [' '.join(preprocess_text(
    entry['product_name'], language='turkish')) for entry in dataset_2]
processed_data_3 = [' '.join(preprocess_text(
    entry['movie'], language='turkish')) for entry in dataset_3]

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(processed_data_1 + processed_data_2 + processed_data_3)

sequences_1 = tokenizer.texts_to_sequences(processed_data_1)
sequences_2 = tokenizer.texts_to_sequences(processed_data_2)
sequences_3 = tokenizer.texts_to_sequences(processed_data_3)

max_seq_length_1 = max(len(x) for x in sequences_1)
max_seq_length_2 = max(len(y) for y in sequences_2)
max_seq_length_3 = max(len(z) for z in sequences_3)

data_1 = pad_sequences(sequences_1, maxlen=max_seq_length_1)
data_2 = pad_sequences(sequences_2, maxlen=max_seq_length_2)
data_3 = pad_sequences(sequences_3, maxlen=max_seq_length_3)


def create_model(input_length):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=input_length),
        LSTM(128, kernel_regularizer=l2(0.01), return_sequences=True),
        Dropout(0.5),
        LSTM(128, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


model_1 = create_model(max_seq_length_1)
model_2 = create_model(max_seq_length_2)
model_3 = create_model(max_seq_length_3)

labels_1 = np.array([i % 2 for i in range(len(data_1))])
labels_2 = np.array([i % 2 for i in range(len(data_2))])
labels_3 = np.array([i % 2 for i in range(len(data_3))])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)

history_1 = model_1.fit(data_1, labels_1, epochs=10, batch_size=64,
                        validation_split=0.2, callbacks=[early_stopping])
history_2 = model_2.fit(data_2, labels_2, epochs=10, batch_size=64,
                        validation_split=0.2, callbacks=[early_stopping])
history_3 = model_3.fit(data_3, labels_3, epochs=10, batch_size=64,
                        validation_split=0.2, callbacks=[early_stopping])


def evaluate_model(model, data, labels):
    predictions = (model.predict(data) > 0.5).astype("int32")
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return precision, recall, f1


precision_1, recall_1, f1_1 = evaluate_model(model_1, data_1, labels_1)
precision_2, recall_2, f1_2 = evaluate_model(model_2, data_2, labels_2)
precision_3, recall_3, f1_3 = evaluate_model(model_3, data_3, labels_3)

print("Model 1 - Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(
    precision_1, recall_1, f1_1))
print("Model 2 - Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(
    precision_2, recall_2, f1_2))
print("Model 3 - Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(
    precision_3, recall_3, f1_3))

example_entry_1 = dataset_1[0]
example_entry_2 = dataset_2[0]
example_entry_3 = dataset_3[0]

print("Example Entry from Dataset 1:", example_entry_1)
print("Example Entry from Dataset 2:", example_entry_2)
print("Example Entry from Dataset 3:", example_entry_3)

print("Length of data_1:", len(data_1))
print("Length of labels_1:", len(labels_1))
print("Length of data_2:", len(data_2))
print("Length of labels_2:", len(labels_2))
print("Length of data_3:", len(data_3))
print("Length of labels_3:", len(labels_3))


def plot_history(history, title):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'],
             label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history.history['precision'], label='Training Precision')
    plt.plot(epochs, history.history['val_precision'],
             label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title(f'{title} - Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history.history['recall'], label='Training Recall')
    plt.plot(epochs, history.history['val_recall'], label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title(f'{title} - Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(history_1, 'Model 1')
plot_history(history_2, 'Model 2')
plot_history(history_3, 'Model 3')
