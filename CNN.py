import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

# Directory containing audio files
directory = 'C:\\Users\\Abhinav\\vf_data'

# Load the audio file
def load_audio(audio_path, sr=44100):
    return librosa.load(audio_path, sr=sr)


# Compute a spectrogram of the audio
def compute_spectrogram(y, sr):
    return librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512)


# Extract spectrogram features
def extract_spectrogram_features(audio_path, max_frames=500):
    y, sr = load_audio(audio_path)
    spectrogram = compute_spectrogram(y, sr)

    # If the spectrogram is shorter than max_frames, pad it with zeros
    if spectrogram.shape[1] < max_frames:
        pad_width = max_frames - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    # If the spectrogram is longer than max_frames, truncate it
    if spectrogram.shape[1] > max_frames:
        spectrogram = spectrogram[:, :max_frames]

    return np.expand_dims(spectrogram, -1)  # Add an extra dimension to the end of the array



def data_generator(features, labels, batch_size):
    batch_features = []
    batch_labels = []

    while True:  # this loop will keep looping over the data indefinitely
        for index, feature in enumerate(features):
            label = labels[index]

            batch_features.append(feature)
            batch_labels.append(label)

            if len(batch_features) >= batch_size:
                yield np.array(batch_features), np.array(batch_labels)
                batch_features = []
                batch_labels = []



# Process multiple audio files

def process_audio_files(directory):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio_path = os.path.join(directory, filename)
            feature = extract_spectrogram_features(audio_path)
            features.append(feature)
            label_str = filename.split('_')[0]  # Assuming the label is the first part of the filename
            
            if label_str == '0.5':
                label = 1
            elif label_str == '1.0':
                label = 2
            else:
                label = 0

            labels.append(label)
    return features, labels


# Define the CNN model
# Define the CNN model
def define_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax')) # 3 is the number of categories
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the CNN on multiple audio files
def train_model_on_multiple_files(directory, batch_size=32):
    features, labels = process_audio_files(directory)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Get the length of the shortest time sequence in your data
    min_length = min(feature.shape[1] for feature in features_train)

    # Manually pad or truncate your spectrograms to min_length
    for i in range(len(features_train)):
        feature = features_train[i]
        if feature.shape[1] > min_length:
            # If the spectrogram is longer than min_length, truncate it
            features_train[i] = feature[:, :min_length]
        elif feature.shape[1] < min_length:
            # If the spectrogram is shorter than min_length, pad it with zeros
            padding = np.zeros((128, min_length - feature.shape[1]))
            features_train[i] = np.concatenate([feature, padding], axis=1)

    # Do the same for features_test
    for i in range(len(features_test)):
        feature = features_test[i]
        if feature.shape[1] > min_length:
            features_test[i] = feature[:, :min_length]
        elif feature.shape[1] < min_length:
            padding = np.zeros((128, min_length - feature.shape[1]))
            features_test[i] = np.concatenate([feature, padding], axis=1)

    features_train = np.array(features_train).astype('float32')
    features_test = np.array(features_test).astype('float32')

    # normalize to [0, 1] range
    features_train /= np.max(features_train)
    features_test /= np.max(features_test)

    print(features_train[0].shape)

    # Define the model
    model = define_model(features_train[0].shape)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    labels_train = to_categorical(labels_train, num_classes=3)
    labels_test = to_categorical(labels_test, num_classes=3)

    train_generator = data_generator(features_train, labels_train, batch_size)
    test_generator = data_generator(features_test, labels_test, batch_size)

    # Train the model
    history = model.fit(train_generator, steps_per_epoch=len(features_train) // batch_size, epochs=10, validation_data=test_generator, validation_steps=len(features_test) // batch_size)
    fig = go.Figure()

    # Training Loss
    fig.add_trace(go.Scatter(
        y=history.history['loss'],
        mode='lines',
        name='train loss'
    ))

    # Validation Loss
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'],
        mode='lines',
        name='validation loss'
    ))

    # Training Accuracy
    fig.add_trace(go.Scatter(
        y=history.history['accuracy'],
        mode='lines',
        name='train accuracy'
    ))

    # Validation Accuracy
    fig.add_trace(go.Scatter(
        y=history.history['val_accuracy'],
        mode='lines',
        name='validation accuracy'
    ))

    fig.show()




# Run the script
train_model_on_multiple_files(directory)