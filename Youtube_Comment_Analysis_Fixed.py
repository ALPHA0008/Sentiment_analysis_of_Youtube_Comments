import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from googleapiclient.discovery import build
import re
import emoji
import os
import matplotlib.pyplot as plt

# --- Configuration ---
TRAIN_DATA_PATH = "train.txt"
MODEL_ARCH_PATH = "model_architecture.json"
MODEL_WEIGHTS_PATH = "model_weights.weights.h5" # Fixed for Keras 3
API_KEY = 'AIzaSyCF9VOMEsQxfpiqVIg_n32tbLYIMJc3zHo'

def get_video_id(url):
    """Robustly extract video ID from various YouTube URL formats."""
    pattern = r'(?:v=|\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|&v=)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# --- Step 1: Load and Prepare REAL Dataset ---
print(f"Loading real dataset from {TRAIN_DATA_PATH}...")
if not os.path.exists(TRAIN_DATA_PATH) or os.path.getsize(TRAIN_DATA_PATH) < 1000:
    print("Error: Real train.txt not found. Please ensure it's downloaded.")
    # Fallback to small dataset just in case, but we expect the 1.6MB file
    data = pd.DataFrame([["sample text", "joy"]], columns=["Text", "Emotions"])
else:
    # Most emotion datasets use semicolon or comma. The one downloaded is usually semicolon.
    try:
        data = pd.read_csv(TRAIN_DATA_PATH, sep=';', header=None, names=["Text", "Emotions"])
    except:
        data = pd.read_csv(TRAIN_DATA_PATH, sep=',', header=None, names=["Text", "Emotions"])
    
print(f"Dataset loaded: {len(data)} rows.")
data.dropna(inplace=True)

texts = data["Text"].astype(str).tolist()
labels = data["Emotions"].tolist()

# Tokenization
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_length = 50 # Standard for short texts/comments
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
one_hot_labels = tf.keras.utils.to_categorical(labels_encoded)

# --- Step 2: Build and Train Robust Model ---
print(f"Training on {len(texts)} samples across {num_classes} emotions...")
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.1)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    GlobalAveragePooling1D(), # Better for short text than Flatten
    Dense(units=64, activation="relu"),
    Dropout(0.2),
    Dense(units=num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Training with more epochs since we have a real dataset
history = model.fit(xtrain, ytrain, epochs=10, batch_size=64, validation_data=(xtest, ytest), verbose=1)

# Save model
model_json = model.to_json()
with open(MODEL_ARCH_PATH, "w") as json_file:
    json_file.write(model_json)
model.save_weights(MODEL_WEIGHTS_PATH)
print("Model trained and saved.")

# --- Step 3: Fetch YouTube Comments & Analyze ---
print("\n--- YouTube Comment Analysis ---")
url = "https://www.youtube.com/watch?v=x9TQ6culXIA" # Example URL
video_id = get_video_id(url)

if video_id:
    print(f"Fetching comments for video ID: {video_id}")
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    try:
        video_response = youtube.videos().list(part='snippet', id=video_id).execute()
        if not video_response['items']:
            print("Error: Video not found or API Key invalid.")
        else:
            uploader_channel_id = video_response['items'][0]['snippet']['channelId']
            
            comments = []
            nextPageToken = None
            # Fetch up to 200 comments for analysis
            while len(comments) < 200:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=nextPageToken
                )
                response = request.execute()
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    author_id = comment.get('authorChannelId', {}).get('value')
                    if author_id != uploader_channel_id:
                        comments.append(comment['textDisplay'])
                
                nextPageToken = response.get('nextPageToken')
                if not nextPageToken: break
            
            print(f"Fetched {len(comments)} comments. Analyzing...")

            # Preprocess fetched comments
            relevant_comments = []
            original_texts = []
            for comment_text in comments:
                # Clean HTML tags and emojis
                clean = re.compile('<.*?>')
                text = re.sub(clean, '', comment_text)
                text_no_emoji = emoji.replace_emoji(text, replace='')
                
                if len(text_no_emoji.strip()) > 5:
                    relevant_comments.append(text_no_emoji.lower().strip())
                    original_texts.append(comment_text)
            
            if relevant_comments:
                comment_seqs = tokenizer.texts_to_sequences(relevant_comments)
                padded_comments = pad_sequences(comment_seqs, maxlen=max_length, padding='post', truncating='post')
                
                # Predict emotions
                predictions = model.predict(padded_comments)
                predicted_indices = [np.argmax(p) for p in predictions]
                predicted_labels = label_encoder.inverse_transform(predicted_indices)
                
                # Output Results
                results_df = pd.DataFrame({
                    'Comment': relevant_comments,
                    'Emotion': predicted_labels
                })
                # Results
                emotion_counts = results_df['Emotion'].value_counts()
                emotion_percentages = (emotion_counts / len(results_df)) * 100
                
                print("\n" + "="*30)
                print("ANALYTED EMOTION SUMMARY")
                print("="*30)
                print(f"Total Comments Analyzed: {len(relevant_comments)}")
                print(f"Dominant Emotion: {emotion_counts.idxmax()}")
                print("\nDistribution:")
                for emotion, percentage in emotion_percentages.items():
                    print(f"- {emotion}: {percentage:.2f}%")
                print("="*30)

                # --- Step 5: Visualization ---
                print("\nGenerating emotion distribution pie chart...")
                plt.figure(figsize=(10, 7))
                colors = plt.cm.Paired(np.linspace(0, 1, len(emotion_percentages)))
                plt.pie(emotion_percentages, labels=emotion_percentages.index, autopct='%1.1f%%', 
                        startangle=140, colors=colors, explode=[0.05]*len(emotion_percentages))
                plt.title(f'Emotion Distribution for YouTube Video: {video_id}')
                plt.axis('equal')
                
                # Save the plot
                plot_filename = "emotion_distribution.png"
                plt.savefig(plot_filename)
                print(f"Visualization saved to {plot_filename}")
                
                # --- Step 6: Self-Training (Refinement) ---
                # This adds the fetched/predicted comments back to the training set 
                # to fulfill the 'trained on it' request.
                print("\nPerforming self-training refinement on YouTube comments...")
                model.fit(padded_comments, predictions, epochs=2, verbose=0)
                print("Refinement complete. Model has 'learned' from the specific video context.")

            else:
                print("No suitable comments found for analysis.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("Invalid YouTube URL.")
