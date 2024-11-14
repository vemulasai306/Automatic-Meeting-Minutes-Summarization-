

import os
import librosa
import soundfile as sf
import speech_recognition as sr
import streamlit as st
import string
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
def clean_filename(filename):
    # Simplify the filename by removing special characters and spaces
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned_filename = ''.join(c for c in filename if c in valid_chars)
    return cleaned_filename.replace(" ", "_")

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    try:
        audio_data, sample_rate = librosa.load(mp3_file_path, sr=None)
        sf.write(wav_file_path, audio_data, sample_rate)
        print(f"Converted {mp3_file_path} to {wav_file_path}")
        return wav_file_path
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    cleaned_filename = clean_filename(audio_file.name)
    wav_file = None  # Initialize wav_file to avoid UnboundLocalError

    # Convert from mp3 to wav if necessary
    if audio_file.name.endswith('.mp3'):
        audio_data = audio_file.read()  # Read the audio file data
        wav_file = f"{cleaned_filename}.wav"
        with open(wav_file, 'wb') as f:
            f.write(audio_data)  # Write it to a file
        converted_wav = convert_mp3_to_wav(wav_file, wav_file)
        if not converted_wav:
            return None
    else:
        wav_file = audio_file  # Use the original file if it's already .wav

    try:
        # Proceed with transcription
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio")
        return None
    except sr.RequestError as e:
        st.error(f"Request error: {e}")
        return None
    finally:
        # Clean up the temporary wav file only if it was created
        if wav_file != audio_file and wav_file is not None:
            os.remove(wav_file)


from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text):
    # Load pre-trained BART model and tokenizer for summarization
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Get model outputs
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

    # Decode the token IDs to generate summary text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def save_summary(summary, filename="meeting_minutes.txt"):
    with open(filename, 'w') as file:
        file.write(summary)

# Function to evaluate summary quality
def evaluate_summary(transcribed_text, summary_text):
    vectorizer = CountVectorizer(binary=True, stop_words='english')
    X = vectorizer.fit_transform([transcribed_text, summary_text])
    transcribed_vector, summary_vector = X.toarray()
    
    precision = precision_score(summary_vector, transcribed_vector)
    recall = recall_score(summary_vector, transcribed_vector)
    f1 = f1_score(summary_vector, transcribed_vector)
    accuracy = np.mean(summary_vector == transcribed_vector)
    
    return precision, recall, f1, accuracy
def validate_transcription(transcribed_text, original_minutes):
    # Compare the transcribed text with the original minutes
    # For simplicity, we can use basic text similarity (Jaccard similarity, Cosine similarity, etc.)
    

    def jaccard_similarity(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        return len(a.intersection(b)) / len(a.union(b))

    similarity_score = jaccard_similarity(transcribed_text, original_minutes)
    return similarity_score

import streamlit as st

st.title("Automatic Meeting Minutes Summarization and Evaluation Tool")

# File upload
audio_file = st.file_uploader("Upload Meeting Audio", type=["wav", "mp3"])

# Set up session state variables to manage progress
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'transcription_done' not in st.session_state:
    st.session_state.transcription_done = False
if 'summary_done' not in st.session_state:
    st.session_state.summary_done = False

# Start processing if an audio file is uploaded
# Start processing if an audio file is uploaded
if audio_file is not None:
    # Perform transcription
    if not st.session_state.transcription_done:
        transcribed_text = transcribe_audio(audio_file)
        
        # Check if transcription is successful
        if transcribed_text:
            st.session_state.transcribed_text = transcribed_text
            st.session_state.transcription_done = True
            st.success("Transcription completed successfully. Now performing summarization...")
        else:
            st.error("No transcription found. Please try uploading a different audio file.")

    # Perform summarization if transcription is completed
    if st.session_state.transcription_done and not st.session_state.summary_done:
        summary = summarize_text(st.session_state.transcribed_text)
        st.session_state.summary = summary
        st.session_state.summary_done = True
        st.success("Summarization completed successfully.")

    # Display summary and evaluation metrics if summarization is done
    if st.session_state.summary_done:
        st.subheader("Summary of Transcription")
        st.write(st.session_state.summary)
        
        # Add a download button for the summary
        st.download_button(
            label="Download Summary as Text File",
            data=st.session_state.summary,
            file_name="summary.txt",
            mime="text/plain"
        )
        
        # Evaluate summary against transcribed text
        precision, recall, f1, accuracy = evaluate_summary(st.session_state.transcribed_text, st.session_state.summary)
        similarity= validate_transcription(st.session_state.transcribed_text, st.session_state.summary)
        st.subheader("Evaluation Metrics")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Similarity: {similarity:.2f}")
