import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

# Download NLTK data
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


sample_size = 500  # Increased sample size

# Load datasets
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

texts = processed_data_1 + processed_data_2 + processed_data_3
labels_1 = np.array([i % 2 for i in range(len(processed_data_1))])
labels_2 = np.array([i % 2 for i in range(len(processed_data_2))])
labels_3 = np.array([i % 2 for i in range(len(processed_data_3))])
labels = np.concatenate([labels_1, labels_2, labels_3])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, lower=True, oov_token='UNK')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42)

# Define the LSTM model with increased regularization and gradient clipping
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=100))
model.add(SpatialDropout1D(0.6))  # Increased dropout
model.add(LSTM(64, return_sequences=True, dropout=0.6,
          recurrent_dropout=0.6))  # Increased LSTM units
# Added another LSTM layer
model.add(LSTM(32, dropout=0.6, recurrent_dropout=0.6))
model.add(BatchNormalization())
# L2 regularization
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

# Compile the model with Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    learning_rate=1e-4), metrics=['accuracy'])  # Reduced learning rate

# Increase epochs and adjust batch size
epochs = 50  # Increased epochs
batch_size = 32  # Adjusted batch size

# Adding early stopping, model checkpoint, and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               min_delta=0.0001, restore_best_weights=True)  # Adjusted patience
model_checkpoint = ModelCheckpoint(
    'best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)  # Learning rate scheduler

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.3, callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Plotting training history


def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_training_history(history)

# Example entries
example_entry_1 = dataset_1[0]
example_entry_2 = dataset_2[0]
example_entry_3 = dataset_3[0]

print("Example Entry from Dataset 1:", example_entry_1)
print("Example Entry from Dataset 2:", example_entry_2)
print("Example Entry from Dataset 3:", example_entry_3)
