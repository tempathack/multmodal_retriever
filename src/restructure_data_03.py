import os
import shutil
import pandas as pd
import textract
import os
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from utils.utils import *

CT=0
base_folder = 'preprocessed_data'
text_folder = os.path.join(base_folder, 'text_files')
images_folder = os.path.join(base_folder, 'images')

os.makedirs(text_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)
def extract_text_from_pdf(pdf_path):
    """
    Given a PDF, this function checks if it's a scanned PDF or not.
    If it's scanned, it uses OCR to extract text.
    Otherwise, it directly extracts text and also extracts images.
    """
    # Initialize PDF reader
    pdf_reader = PdfReader(pdf_path)

    # Extract text using PyPDF2
    all_text = ""
    images = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text:  # If text is found, it's not a scanned PDF
            all_text += text
        else:
            # If text is not found, assume the page is an image (scanned)
            print(f"Page {page_num + 1} seems to be scanned. Using OCR.")
            images.extend(convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1))

    if all_text:
        print("PDF is not scanned. Returning extracted text.")
        return all_text, []
    else:
        print("PDF is scanned. Performing OCR on extracted images.")
        # Perform OCR on the images
        extracted_text = ""
        for img in images:
            extracted_text += pytesseract.image_to_string(img)
        return extracted_text, images


def save_images(images, output_folder):
    """Save extracted images to the specified folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, img in enumerate(images):
        img.save(os.path.join(output_folder, f"image_{i + 1}.jpg"))

def bytes_to_text(byte_data, encoding='utf-8'):
    """
    Convert bytes to text using the specified encoding.

    Args:
        byte_data (bytes): The byte data to convert.
        encoding (str): The encoding to use (default is 'utf-8').

    Returns:
        str: The decoded text.
    """
    try:
        text = byte_data.decode(encoding)
        return text
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding bytes: {e}")
# Function to read .docx file
def read_docx(file_path):
    try:
        text = textract.process(file_path)
        return text
    except Exception as e:
        return f"Error reading .docx file: {e}"
# Function to read .pdf file

# Function to read and sort files
def sort_files(files):
    for i,file_path in enumerate(files):
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.docx':
            text = bytes_to_text(read_docx(file_path))

            _flg = is_gibberish(text)
            if _flg:
                CT=+1


            text_filename = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
            text_path = os.path.join(text_folder, text_filename)
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            print(f"Saved text from {file_path} to {text_path}")

        elif file_ext == '.pdf':

            file_name=os.path.basename(file_path)
            file_name.replace('.pdf','jpeg')

            text, images = extract_text_from_pdf(file_path)

            _flg=is_gibberish(text)
            if _flg:
                CT=+1



            if text:
                text_filename = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
                text_path = os.path.join(text_folder, text_filename)
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)

            if images:


                for img in images:
                    img.save(os.path.join(images_folder,f'img_{i}'+file_name), format='jpeg')

        elif file_ext in ['.jpg', '.png']:
            new_path = os.path.join(images_folder, os.path.basename(file_path))
            shutil.copy(file_path, new_path)
            print(f"Copied {file_path} to {images_folder}")

        else:
            print(f"Unsupported file format: {file_ext}")


# Read file paths from CSV
df = pd.read_csv('./file_tracking/tracked_files.csv')
files = df['File Path'].tolist()  # Assuming the column name is 'File Path'

# Sort files
sort_files(files)
print(f""" {CT/len(files)*100} % of files are gibbrish """)
