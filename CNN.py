import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Directory containing audio files
directory = 'C:\\Users\\Abhinav\\vf_data'

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
            label_str = filename.split('_')[0]  # Assuming the label is the first part of the filename
            if label_str == '0.5':
                label = 0.5
            else:
                label = int(label_str)
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
    model.add(Dense(3, activation='softmax'))
    return model

# Train the CNN on multiple audio files
def train_model_on_multiple_files(directory):
    features, labels = process_audio_files(directory)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
   # Set the desired length for the spectrogram features
    desired_length = 7925
  

    # Truncate or pad the spectrogram features to the desired length
    features_train = pad_sequences(features_train, maxlen=desired_length, padding='post', truncating='post')
    features_test = pad_sequences(features_test, maxlen=desired_length, padding='post', truncating='post')

    features_train = np.array(features_train)
    features_train = features_train.astype('int16')
    features_test = np.array(features_test)
    features_test = features_test.astype('int16')
    # Define the model
    model = define_model(features_train[0].shape)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    labels_train = to_categorical(labels_train, num_classes=3)
    labels_test = to_categorical(labels_test, num_classes=3)

    # Train the model
    history = model.fit(features_train, labels_train, epochs=10, batch_size=32, validation_data=(features_test, labels_test))
    
    # Train the model
    model.fit(features_train, labels_train, epochs=10, batch_size=32, validation_data=(features_test, labels_test))

# Run the script
train_model_on_multiple_files(directory)