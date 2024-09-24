import os
import numpy as np
from sklearn.cluster import KMeans
from skimage import io, transform
from skimage.filters import sobel
from PIL import Image

# Increase the decompression bomb limit
Image.MAX_IMAGE_PIXELS = 933120000  # Sets a new, higher limit


# Function to extract edge-based complexity feature
def image_complexity(image_path):
    try:
        # Load image
        img = io.imread(image_path, as_gray=True)

        # Resize if the image is too large
        max_pixels = 933120000  # New limit
        if img.size > max_pixels:
            scale = (max_pixels / img.size) ** 0.5
            new_shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))
            img = transform.resize(img, new_shape, anti_aliasing=True)

        # Apply edge detection
        edges = sobel(img)
        # Sum of edge intensities as a measure of complexity
        complexity = np.sum(edges)
        return complexity
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


# Get image paths
image_paths = [os.path.join('./preprocessed_data/images', pth) for pth in os.listdir('./preprocessed_data/images')]

# Extract features (complexity scores)
features = []
valid_image_paths = []

for img in image_paths:
    complexity = image_complexity(img)
    if complexity is not None:
        features.append(complexity)
        valid_image_paths.append(img)

features = np.array(features).reshape(-1, 1)

# Apply K-Means clustering (2 clusters: simple, complex)
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(features)

# Output clusters
for i, img in enumerate(valid_image_paths):
    print(f"Image: {img} is in Cluster: {clusters[i]}")
    if clusters[i]==0:
        os.remove(img)