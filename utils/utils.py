import re
import time
import os
from langdetect import detect
import pandas as pd
import plotly.express as px
from  datetime import datetime
from pathlib import Path
common_german_words = set([
    "und", "der", "die", "das", "ist", "du", "ich", "nicht", "es", "sie", "wir", "ein", "er", "in", "zu", "haben",
    "auch", "ja", "nein", "mit", "wie", "auf", "für", "den", "von", "dem", "sind", "dass", "ich", "mir", "mich", "sein",
    "bei", "um", "aber", "hier", "da", "wenn", "nur", "dann", "war", "noch", "nach", "wird", "kann", "können", "wir"
    # Add more words as needed
])


def get_file_modified_time(file_path):
    # Get file metadata
    file_info = Path(file_path)

    # Get the modification time (in seconds since epoch)
    modified_time = file_info.stat().st_mtime

    # Convert to a human-readable format
    modified_time = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')

    return modified_time
def is_gibberish(text):
    # Remove non-alphabetic characters, except for spaces and common German punctuation
    cleaned_text = re.sub(r'[^A-Za-zäöüÄÖÜß ]+', '', text)

    # Check if the detected language is German
    try:
        detected_language = detect(cleaned_text)
        if detected_language != 'de':
            return True
    except:
        return True

    # Split the text into words
    words = cleaned_text.split()

    # Check if a significant number of words are not in the common German words list
    non_german_word_count = sum(1 for word in words if word.lower() not in common_german_words)

    # Define a threshold for gibberish detection (e.g., more than half non-German words)
    threshold = 0.5
    if len(words) > 0 and (non_german_word_count / len(words)) > threshold:
        return True

    return False
def sanitize_string(input_string):
    # Remove non-printable/control characters using regex
    sanitized_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', input_string)

    # Optionally, replace multiple newlines or spaces with a single space
    sanitized_string = re.sub(r'\s+', ' ', sanitized_string).strip()

    return sanitized_string
def sanitize_input(doc_list):
    # Remove control characters
    sanitized_docs = [re.sub(r'[\x00-\x1f\x7f-\x9f]', '', doc) for doc in doc_list]
    # Remove empty or whitespace-only docs
    sanitized_docs = [doc for doc in sanitized_docs if doc.strip()]
    # Ensure UTF-8 encoding
    sanitized_docs = [doc.encode('utf-8', 'ignore').decode('utf-8') for doc in sanitized_docs]
    return sanitized_docs
def track_files_in_directory(directory):
    file_data = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)  # Get file size in bytes
            file_ext = os.path.splitext(file)[1]  # Get file extension
            file_creation_time = time.ctime(os.path.getctime(file_path))  # Get file creation time

            # Append file details to list
            file_data.append({
                'File Name': file,
                'File Path': file_path,
                'File Size (Bytes)': file_size,
                'File Extension': file_ext,
                'Creation Time': file_creation_time,
                'Manipulation Time':get_file_modified_time(file_path)
            })

    # Create a DataFrame from the list
    df = pd.DataFrame(file_data)
    return df
def get_file_sizes(directory):
    file_data = []

    # Walk through the directory and collect file information
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)  # Get file size in bytes
            relative_path = os.path.relpath(file_path, directory)  # Get relative file path

            file_data.append({
                'File Name': file,
                'File Path': relative_path,
                'File Size (Bytes)': file_size,
                'Directory': os.path.basename(root)  # Directory name
            })

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(file_data)
    return df
def create_treemap(df):
    # Create a Plotly treemap with file sizes
    fig = px.treemap(
        df,
        path=['Directory', 'File Name'],  # Hierarchical view
        values='File Size (Bytes)',  # Size of boxes based on file size
        title='File Sizes Treemap',
        color='File Size (Bytes)',  # Color based on file size
        hover_data={'File Size (Bytes)': ':.2f'}  # Show file size on hover
    )

    # Show the figure
    #fig.write_html('./Laufwerk_structure.html')


def create_word_doc(inputs, answers, documents,al_docs):
    from collections import defaultdict
    from docx import Document
    from io import BytesIO
    doc = Document()

    # Add title
    doc.add_heading('Session Information', level=1)

    # Add Input Section
    doc.add_heading('Input Prompts', level=2)
    for i, item in enumerate(inputs, 1):
        doc.add_paragraph(f"{i}. {item}")

    # Add Answer Section
    doc.add_heading('Generated Answers', level=2)
    for i, item in enumerate(answers, 1):
        doc.add_paragraph(f"{i}. {item}")

    # Add Document Section
    doc.add_heading('Most Relevant Documents', level=2)
    for i, item in enumerate(documents, 1):
        doc.add_paragraph(f"{i}. {item}")

    doc.add_heading('All associated Documents', level=2)
    for i, item in enumerate(al_docs, 1):
        doc.add_paragraph(f"{i}. {str(item)}")

    # Save to BytesIO object
    byte_io = BytesIO()
    doc.save(byte_io)
    byte_io.seek(0)

    return byte_io