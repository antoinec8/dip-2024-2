# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import cv2 # Usado apenas para carregar as imagens
import numpy as np
import matplotlib.pyplot as plt # Usando apenas para exibir as imagens orginais e modificadas

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape

    # Translada a imagem para a direita e para baixo
    def translate_image(shift_x=10, shift_y=10):
        out = np.zeros_like(img)
        out[shift_y:h, shift_x:w] = img[:h-shift_y, :w-shift_x]
        return out

    # Rotaciona a imagem 90 graus no sentido horário
    def rotate_image():
        return np.rot90(img, -1)

    # Estica horizontalmente a imagem, aumentando a largura em 1.5x
    def stretch_image(scale=1.5):
        new_w = int(w * scale)
        out = np.zeros((h, new_w), dtype=img.dtype)
        for i in range(new_w):
            orig_x = int(i / scale)
            out[:, i] = img[:, orig_x]
        return out

    # Espelha horizontalmente a imagem (flip vertical)
    def mirror_image():
        return img[:, ::-1]

    # Aplica uma distorção de barril (barrel distortion) simples.
    # Para cada pixel de saída, calcula as coordenadas de origem com base em uma função radial.
    def barrel_distort(k=-1e-5):
        y_indices, x_indices = np.indices((h, w))
        cx, cy = w / 2.0, h / 2.0
        dx = x_indices - cx
        dy = y_indices - cy
        r = np.sqrt(dx**2 + dy**2)
        factor = 1 + k * r**2
        x_src = cx + dx / factor
        y_src = cy + dy / factor
        x_src_nn = np.clip(np.round(x_src).astype(int), 0, w - 1)
        y_src_nn = np.clip(np.round(y_src).astype(int), 0, h - 1)
        return img[y_src_nn, x_src_nn]

    return {
        "translated": translate_image(),
        "rotated": rotate_image(),
        "stretched": stretch_image(),
        "mirrored": mirror_image(),
        "distorted": barrel_distort()
    }

img = cv2.imread('img\lena.png', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

out = apply_geometric_transformations(img)

fig, axs = plt.subplots(1, 6, figsize=(18, 3))
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original")
axs[1].imshow(out["translated"], cmap='gray')
axs[1].set_title("Translated")
axs[2].imshow(out["rotated"], cmap='gray')
axs[2].set_title("Rotated")
axs[3].imshow(out["stretched"], cmap='gray')
axs[3].set_title("Stretched")
axs[4].imshow(out["mirrored"], cmap='gray')
axs[4].set_title("Mirrored")
axs[5].imshow(out["distorted"], cmap='gray')
axs[5].set_title("Distorted")

for ax in axs:
    ax.axis('off')

plt.show()