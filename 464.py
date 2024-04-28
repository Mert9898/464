import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
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
    "hkust-nlp/deita-quality-scorer-data", split='validation')  # Corrected split
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
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model_1 = create_model(max_seq_length_1)
model_2 = create_model(max_seq_length_2)
model_3 = create_model(max_seq_length_3)


labels_1 = np.array([i % 2 for i in range(len(data_1))])
labels_2 = np.array([i % 4 for i in range(len(data_2))])
labels_3 = np.array([i % 3 for i in range(len(data_3))])


model_1.fit(data_1, labels_1, epochs=10, validation_split=0.2)
model_2.fit(data_2, labels_2, epochs=10, validation_split=0.2)
model_3.fit(data_3, labels_3, epochs=10, validation_split=0.2)

info = load_dataset("hkust-nlp/deita-quality-scorer-data")
print(info)

example_entry = dataset_1[0]
print(example_entry)

print(dataset_1[0])
print(dataset_2[0])
print(dataset_3[0])
print("Length of data_1:", len(data_1))
print("Length of labels_1:", len(labels_1))
print("Length of data_2:", len(data_2))
print("Length of labels_2:", len(labels_2))
print("Length of data_3:", len(data_3))
print("Length of labels_3:", len(labels_3))
