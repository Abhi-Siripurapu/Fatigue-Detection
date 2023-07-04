import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Directory containing audio files
directory = 'path_to_your_audio_files'

# Load the audio file
def load_audio(audio_path):
    return librosa.load(audio_path)

# Compute a spectrogram of the audio
def compute_spectrogram(y, sr):
    return librosa.feature.melspectrogram(y=y, sr=sr)

# Extract spectrogram features
def extract_spectrogram_features(audio_path):
    y, sr = load_audio(audio_path)
    spectrogram = compute_spectrogram(y, sr)
    return spectrogram

# Process multiple audio files
def process_audio_files(directory):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio_path = os.path.join(directory, filename)
            feature = extract_spectrogram_features(audio_path)
            features.append(feature)
            label = filename.split('_')[0]  # Assuming the label is the first part of the filename
            labels.append(label)
    return features, labels

# Define the CNN model
def define_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Train the CNN on multiple audio files
def train_model_on_multiple_files(directory):
    features, labels = process_audio_files(directory)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Reshape the features for the CNN
    features_train = np.expand_dims(features_train, axis=-1)
    features_test = np.expand_dims(features_test, axis=-1)
    
    # Define the model
    model = define_model(features_train[0].shape)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(features_train, labels_train, epochs=10, batch_size=32, validation_data=(features_test, labels_test))

# Run the script
train_model_on_multiple_files(directory)