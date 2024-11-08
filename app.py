

import os
import librosa
import soundfile as sf
import speech_recognition as sr
import streamlit as st
import string
import os






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


# In[2]:


import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and clean text
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [token.text for token in doc if token.text not in stop_words and token.text not in punctuation]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(tokens)
    return cleaned_text


# In[3]:


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




# In[4]:


def save_summary(summary, filename="meeting_minutes.txt"):
    with open(filename, 'w') as file:
        file.write(summary)


# In[5]:


from fpdf import FPDF

def save_summary_as_pdf(summary, filename="meeting_minutes.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(filename)


# In[6]:


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


# In[7]:


import streamlit as st

st.title("Automatic Meeting Minutes Summarization Tool")

# File upload
audio_file = st.file_uploader("Upload Meeting Audio", type=["wav", "mp3"])

# In the Streamlit section where you call transcribe_audio
if audio_file is not None:
    # Transcribe and summarize
    transcribed_text = transcribe_audio(audio_file)
    
    if transcribed_text is not None:
        # Only preprocess and summarize if transcription was successful
        preprocessed_text = preprocess_text(transcribed_text)
        summary = summarize_text(preprocessed_text)
        
        # Display summary
        st.write(summary)
        
        # Save summary to file
        save_summary(summary)
    else:
        st.error("No Transcription Found. Please try uploading a different audio file.")



# In[ ]:

