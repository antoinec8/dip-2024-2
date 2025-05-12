import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('img/horse.png', cv2.IMREAD_GRAYSCALE)

# Converter para binária
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(15, 3))
plt.subplot(1, 5, 1)
plt.imshow(binary, cmap='gray')
plt.title("Original (Horse)")
plt.axis('off')

# 1. Esqueletonização
def skeletonize(img):
    img = img.copy() // 255
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel * 255

skeleton = skeletonize(binary)
plt.subplot(1, 5, 2)
plt.imshow(skeleton, cmap='gray')
plt.title("1. Esqueletonização")
plt.axis('off')

# 2. Reconstrução morfológica

# 2.1 Dilatação Geodésica
def dilatacao_geodesica(marker, mask):
    kernel = np.ones((3,3), np.uint8)
    prev = np.zeros_like(marker)
    atual = marker.copy()
    while True:
        dilatada = cv2.dilate(atual, kernel)
        atual = cv2.min(dilatada, mask)
        if np.array_equal(atual, prev):
            break
        prev = atual.copy()
    return atual

marker = cv2.erode(binary, np.ones((7,7), np.uint8))
dil_geo = dilatacao_geodesica(marker, binary)
plt.subplot(1, 5, 3)
plt.imshow(dil_geo, cmap='gray')
plt.title("2.1 Dilat. Geodésica")
plt.axis('off')

# 2.2 Reconstrução por dilatação e erosão
def reconstrucao(marker, mask, method='dilation'):
    kernel = np.ones((3,3), np.uint8)
    prev = np.zeros_like(marker)
    atual = marker.copy()
    while True:
        if method == 'dilation':
            temp = cv2.dilate(atual, kernel)
            atual = cv2.min(temp, mask)
        else:  
            temp = cv2.erode(atual, kernel)
            atual = cv2.max(temp, mask)
        if np.array_equal(atual, prev):
            break
        prev = atual.copy()
    return atual

# Dilatação
marker_dil = cv2.erode(binary, np.ones((7,7), np.uint8))
rec_dil = reconstrucao(marker_dil, binary, method='dilation')

# Erosão
marker_ero = cv2.dilate(binary, np.ones((7,7), np.uint8))
rec_ero = reconstrucao(marker_ero, binary, method='erosion')

fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].imshow(rec_dil, cmap='gray')
ax[0].set_title("2.2 Rec. Dilatação")
ax[0].axis('off')

ax[1].imshow(rec_ero, cmap='gray')
ax[1].set_title("2.2 Rec. Erosão")
ax[1].axis('off')

# 2.3 Abertura por Reconstrução
eroded = cv2.erode(binary, np.ones((7,7), np.uint8))
abertura_rec = reconstrucao(eroded, binary, method='dilation')
plt.figure(figsize=(5,5))
plt.imshow(abertura_rec, cmap='gray')
plt.title("2.3 Abertura por Reconstrução")
plt.axis('off')

# 2.4 Eliminação elementos de borda
def clear_border(img):
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    img_copy = img.copy()
    cv2.floodFill(img_copy, mask, (0,0), 0)
    return img_copy

sem_borda = clear_border(binary)
plt.figure(figsize=(5,5))
plt.imshow(sem_borda, cmap='gray')
plt.title("2.4 Sem bordas")
plt.axis('off')

plt.show()