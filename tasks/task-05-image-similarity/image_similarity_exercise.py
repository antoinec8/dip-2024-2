# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following out:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import cv2 # Usado apenas para carregar as imagens
import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here

    # Calcula o Mean Squared Error (MSE)
    def mse() -> float:
        return np.mean((i1 - i2) ** 2)

    # Calcula o Peak Signal-to-Noise Ratio (PSNR)
    def psnr() -> float:
        error = mse()
        if error == 0:
            return float('inf')
        return 10 * np.log10(1.0 / error)

    # Calcula um SSIM simplificado
    def ssim() -> float:
        mu1 = np.mean(i1)
        mu2 = np.mean(i2)
        sigma1_sq = np.var(i1)
        sigma2_sq = np.var(i2)
        sigma12 = np.mean((i1 - mu1) * (i2 - mu2))
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    # Calcula o Normalized Pearson Correlation Coefficient (NPCC)
    def npcc() -> float:
        mu1 = np.mean(i1)
        mu2 = np.mean(i2)
        numerator = np.sum((i1 - mu1) * (i2 - mu2))
        denominator = np.sqrt(np.sum((i1 - mu1) ** 2) * np.sum((i2 - mu2) ** 2))
        if denominator == 0:
            return 0
        return numerator / denominator

    return {
        "mse": mse(),
        "psnr": psnr(),
        "ssim": ssim(),
        "npcc": npcc()
    }

img1 = cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img\lena.png', cv2.IMREAD_GRAYSCALE)

img1 = img1.astype(np.float32) / 255.0
img2 = img2.astype(np.float32) / 255.0

out = compare_images(img1, img2)

print("Métricas de Similaridade:")
print("MSE:", out["mse"])
print("PSNR:", out["psnr"])
print("SSIM:", out["ssim"])
print("NPCC:", out["npcc"])