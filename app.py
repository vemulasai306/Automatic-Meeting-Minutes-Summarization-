import os
import librosa
import soundfile as sf
import speech_recognition as sr
import streamlit as st
import string
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from fpdf import FPDF

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

def split_audio(audio_path, chunk_duration=30):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Calculate the number of samples per chunk
    chunk_size = sr * chunk_duration
    
    # Split the audio into chunks
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    
    return chunks, sr

def save_chunks(chunks, sr, base_filename="chunk"):
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_filename}_{i}.wav"
        sf.write(chunk_filename, chunk, sr)
        chunk_files.append(chunk_filename)
    return chunk_files

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

    # Split large audio file into smaller chunks
    chunks, sr = split_audio(wav_file, chunk_duration=30)
    chunk_files = save_chunks(chunks, sr, base_filename=cleaned_filename)

    # Initialize the full transcribed text
    full_transcribed_text = ""

    for chunk_file in chunk_files:
        try:
            with sr.AudioFile(chunk_file) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            full_transcribed_text += text + " "
        except sr.UnknownValueError:
            st.error(f"Could not understand the audio in chunk {chunk_file}")
        except sr.RequestError as e:
            st.error(f"Request error with chunk {chunk_file}: {e}")

    # Cleanup temporary chunk files
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    return full_transcribed_text.strip()

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

def save_summary_as_pdf(summary, filename="meeting_minutes.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(filename)

def validate_transcription(transcribed_text, original_minutes):
    # Compare the transcribed text with the original minutes
    # For simplicity, we can use basic text similarity (Jaccard similarity, Cosine similarity, etc.)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def jaccard_similarity(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        return len(a.intersection(b)) / len(a.union(b))

    similarity_score = jaccard_similarity(transcribed_text, original_minutes)
    print(f"Similarity Score: {similarity_score:.2f}")
    return similarity_score


# Streamlit UI

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
        save_summary_as_pdf(summary)
    else:
        st.error("No Transcription Found. Please try uploading a different audio file.")
