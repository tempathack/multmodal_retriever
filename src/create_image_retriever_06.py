import faiss
import json
import torch
from openai import OpenAI
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import os
import numpy as np
import pickle
from typing import List, Union, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import base64
from tqdm import tqdm

# Initialize OpenAI client and CLIP model
client = OpenAI()
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 1. Image Path Retrieval
def get_image_paths(directory: str, number: int = None) -> List[str]:
    image_paths = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):
            image_paths.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return [image_paths[-1]]
            count += 1
    return image_paths


# 2. Feature Extraction
def get_features_from_image_path(image_paths):
    images = [preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    image_input = torch.tensor(np.stack(images))
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    return image_features


# 3. Image Encoding
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')


# 4. Image Querying
def image_query(query, image_path):
    response = client.chat.completions.create(
        model='gpt-4-vision-preview',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


# 5. Data Handling
def find_entry(data, key, value):
    for entry in data:
        if entry.get(key) == value:
            return entry
    return None


# Main execution
if __name__ == "__main__":
    # Setup
    direc = 'image_database/'
    image_paths = get_image_paths(direc)
    image_features = get_features_from_image_path(image_paths)

    # Create FAISS index
    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)

    # Load description data
    data = []
    with open('description.json', 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Example usage
    image_path = 'train1.jpeg'

    # Query image content
    print(image_query('Write a short label of what is shown in this image?', image_path))

    # Find similar images
    image_search_embedding = get_features_from_image_path([image_path])
    distances, indices = index.search(image_search_embedding.reshape(1, -1), 2)
    indices_distances = list(zip(indices[0], distances[0]))
    indices_distances.sort(key=lambda x: x[1], reverse=True)

    # Get similar image and its description
    similar_path = get_image_paths(direc, indices_distances[0][0])[0]
    element = find_entry(data, 'image_path', similar_path)

    # Query with context
    user_query = 'What is the capacity of this item?'
    prompt = f"""
    Below is a user query, I want you to answer the query using the description and image provided.

    user query:
    {user_query}

    description:
    {element['description']}
    """
    print(image_query(prompt, similar_path))