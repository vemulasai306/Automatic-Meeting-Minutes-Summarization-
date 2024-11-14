import os
import librosa
import soundfile as sf
import speech_recognition as sr
import streamlit as st
import string
import torch
import math
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Function to clean filename for valid characters
def clean_filename(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned_filename = ''.join(c for c in filename if c in valid_chars)
    return cleaned_filename.replace(" ", "_")

# Function to convert mp3 file to wav
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    try:
        audio_data, sample_rate = librosa.load(mp3_file_path, sr=None)
        sf.write(wav_file_path, audio_data, sample_rate)
        return wav_file_path
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return None

# Function to split audio into chunks based on duration
def split_audio(audio_file_path, chunk_duration=30):
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    audio_length = librosa.get_duration(y=audio_data, sr=sample_rate)
    num_chunks = math.ceil(audio_length / chunk_duration)
    chunk_paths = []

    for i in range(num_chunks):
        start_sample = i * chunk_duration * sample_rate
        end_sample = min((i + 1) * chunk_duration * sample_rate, len(audio_data))
        chunk_data = audio_data[int(start_sample):int(end_sample)]
        chunk_path = f"chunk_{i}.wav"
        sf.write(chunk_path, chunk_data, sample_rate)
        chunk_paths.append(chunk_path)
    
    return chunk_paths

# Function to transcribe audio using parallel processing for chunks
def transcribe_chunk(chunk_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(chunk_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "Request error"

def transcribe_audio_parallel(chunk_paths):
    with ThreadPoolExecutor() as executor:
        transcriptions = list(executor.map(transcribe_chunk, chunk_paths))
    return '\n'.join(transcriptions)

# Function to transcribe audio file
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    cleaned_filename = clean_filename(audio_file.name)
    wav_file = None

    if audio_file.name.endswith('.mp3'):
        audio_data = audio_file.read()
        wav_file = f"{cleaned_filename}.wav"
        with open(wav_file, 'wb') as f:
            f.write(audio_data)
        converted_wav = convert_mp3_to_wav(wav_file, wav_file)
        if not converted_wav:
            return None
    else:
        wav_file = audio_file

    chunk_paths = split_audio(wav_file)
    transcribed_text = transcribe_audio_parallel(chunk_paths)

    # Clean up chunk files
    for chunk_path in chunk_paths:
        os.remove(chunk_path)

    return transcribed_text if transcribed_text else None



# Function to load BART model with better caching and device selection
@st.cache_resource
def load_bart_model():
    # Check if CUDA (GPU) is available and only load it on GPU if it is
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
    
    return tokenizer, model

# Function to summarize text
def summarize_text(text):
    # Load the model (cached)
    tokenizer, model = load_bart_model()
    
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(model.device)
    
    # Generate summary using the BART model
    summary_ids = model.generate(inputs['input_ids'], max_length=200, num_beams=6, early_stopping=True, length_penalty=2.0)
    
    # Decode and return the summary
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

# Streamlit App
st.title("Automatic Meeting Minutes Summarization and Evaluation Tool")

# File upload
audio_file = st.file_uploader("Upload Meeting Audio", type=["wav", "mp3"])

if audio_file is not None:
    transcribed_text = transcribe_audio(audio_file)
    
    if transcribed_text is not None:
        st.subheader("Transcribed Text")
        st.write(transcribed_text)
        
        summary = summarize_text(transcribed_text)
        st.subheader("Summary of Transcription")
        st.write(summary)
        
        save_summary(summary)
        
        # Evaluate summary against transcribed text
        precision, recall, f1, accuracy = evaluate_summary(transcribed_text, summary)
        st.subheader("Evaluation Metrics")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")
    else:
        st.error("No transcription found. Please try uploading a different audio file.")
