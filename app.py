
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
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=100, num_beams=6, early_stopping=True, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def save_summary(summary, filename="meeting_minutes.txt"):
    with open(filename, 'w') as file:
        file.write(summary)


from fpdf import FPDF

def save_summary_as_pdf(summary, filename="meeting_minutes.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(filename)


def evaluate_summary(transcribed_text, summary_text):
    vectorizer = CountVectorizer(binary=True, stop_words='english')
    X = vectorizer.fit_transform([transcribed_text, summary_text])
    transcribed_vector, summary_vector = X.toarray()
    
    precision = precision_score(summary_vector, transcribed_vector)
    recall = recall_score(summary_vector, transcribed_vector)
    f1 = f1_score(summary_vector, transcribed_vector)
    accuracy = np.mean(summary_vector == transcribed_vector)
    
    return precision, recall, f1, accuracy


import streamlit as st

st.title("Automatic Meeting Minutes Summarization Tool")

# File upload
audio_file = st.file_uploader("Upload Meeting Audio", type=["wav", "mp3"])

# In the Streamlit section where you call transcribe_audio
if audio_file is not None:
    # Transcribe and summarize
    transcribed_text = transcribe_audio(audio_file)
    
    if transcribed_text is not None:
        # Directly pass the raw transcription to the summarization function
        summary = summarize_text(transcribed_text)
        
        # Display summary
        st.write(summary)
        
        # Save summary to file
        save_summary(summary)
        
        # Evaluate summary against transcribed text
        precision, recall, f1, accuracy = evaluate_summary(transcribed_text, summary)
        print(precision, recall, f1, accuracy)
        st.subheader("Evaluation Metrics")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")
    else:
        st.error("No Transcription Found. Please try uploading a different audio file.")
