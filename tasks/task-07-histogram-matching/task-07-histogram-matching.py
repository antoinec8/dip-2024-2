# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage.exposure as exposure
import matplotlib.pyplot as plt

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    # Your implementation here
    # Cria um array float para acumular o resultado
    matched_float = np.zeros_like(source_img, dtype=np.float32)

    # Ajusta o histograma de cada canal RGB individualmente
    for c in range(3):
        # match_histograms retorna array float com a transformação aplicada
        matched_channel = exposure.match_histograms(
            source_img[..., c],
            reference_img[..., c]
        )
        matched_float[..., c] = matched_channel

    # Garante que os valores estejam dentro de [0, 255] e converte para uint8
    matched_img = np.clip(matched_float, 0, 255).astype(np.uint8)
    return matched_img

source_bgr = cv.imread('tasks/task-07-histogram-matching\source.jpg')    
reference_bgr = cv.imread('tasks/task-07-histogram-matching/reference.jpg')

source_rgb = cv.cvtColor(source_bgr, cv.COLOR_BGR2RGB)
reference_rgb = cv.cvtColor(reference_bgr, cv.COLOR_BGR2RGB)

matched = match_histograms_rgb(source_rgb, reference_rgb)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(source_rgb)
plt.title("Source")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reference_rgb)
plt.title("Reference")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(matched)
plt.title("Output")
plt.axis('off')

plt.tight_layout()
plt.show()