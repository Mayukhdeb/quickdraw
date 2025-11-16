from quickdraw import QuickDrawDataset

dataset = QuickDrawDataset(
    split="train",
    image_size=(224, 224),
    cache_dir = "./hf_cache",
    custom_class_names=["face"],
)

pil_images = [
    dataset[i]["image"] for i in range(100)
]


from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

def generate_eigenimages(pil_images, N):
    """
    Generate N Eigenimages from a list of PIL images.

    Args:
        pil_images (list of PIL.Image): Input images (grayscale or RGB).
        N (int): Number of Eigenimages to compute.

    Returns:
        eigen_pil_images (list of PIL.Image): List of N Eigenimages as PIL images.
    """
    if not pil_images:
        return []

    # Convert all images to grayscale and resize to the first image size
    base_size = pil_images[0].size
    gray_images = [img.convert('L').resize(base_size) for img in pil_images]

    # Flatten images into vectors
    img_vectors = np.array([np.array(img).flatten() for img in gray_images])

    # Perform PCA
    pca = PCA(n_components=N)
    pca.fit(img_vectors)
    eigenvectors = pca.components_

    # Reshape eigenvectors back to image dimensions
    eigen_images = [vec.reshape(base_size[1], base_size[0]) for vec in eigenvectors]

    # Normalize for visualization
    eigen_images_norm = [(img - img.min()) / (img.max() - img.min()) * 255 for img in eigen_images]
    eigen_pil_images = [Image.fromarray(img.astype(np.uint8)) for img in eigen_images_norm]

    return eigen_pil_images

# Generate top 5 Eigenimages
eigen_images = generate_eigenimages(pil_images, 5)

# Save or display them
for i, img in enumerate(eigen_images):
    img.save(f'eigenimage_{i}.png')
