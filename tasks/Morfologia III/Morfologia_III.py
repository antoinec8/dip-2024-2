import cv2
import numpy as np
import matplotlib.pyplot as plt

# 4. Operações morfológicas em escala de cinza

def dilate(img, ksize=3):
    """4.1 Dilatação"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(img, se)

def erode(img, ksize=3):
    """4.1 Erosão"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(img, se)

def opening(img, ksize=3):
    """4.2 Abertura = erosão seguida de dilatação"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, se)

def closing(img, ksize=3):
    """4.2 Fechamento = dilatação seguida de erosão"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

def reconstruction_by_dilation(marker, mask):
    """4.3 Reconstrução morfológica por dilatação"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    prev = np.zeros_like(marker)
    curr = marker.copy()
    while True:
        curr = np.minimum(cv2.dilate(curr, se), mask)
        if np.array_equal(curr, prev):
            break
        prev = curr.copy()
    return curr

def reconstruction_by_erosion(marker, mask):
    """4.3 Reconstrução morfológica por erosão"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    prev = np.zeros_like(marker)
    curr = marker.copy()
    while True:
        curr = np.maximum(cv2.erode(curr, se), mask)
        if np.array_equal(curr, prev):
            break
        prev = curr.copy()
    return curr

def morphological_gradient(img, ksize=3):
    """4.4 Gradiente morfológico = dilatação – erosão"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)

def top_hat(img, ksize=3):
    """4.5 Top-hat = imagem original – abertura"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)

def bottom_hat(img, ksize=3):
    """4.5 Bottom-hat = fechamento – imagem original"""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)

def granulometry(img, max_radius=10, step=2):
    """
    4.6 Granulometria simples:
    Aplica aberturas com elementos elípticos crescentes e mede área.
    Retorna lista de raios e lista de áreas pós-abertura.
    """
    radii = list(range(1, max_radius+1, step))
    areas = []
    for r in radii:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
        areas.append(np.count_nonzero(opened))
    return radii, areas

def texture_segmentation(img, ksize=15, thresh=20):
    """
    4.7 Segmentação de texturas via top-hat:
    - Remove estruturas largas (abertura)
    - Diferença = componente de textura
    - Binariza por threshold
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    texture = cv2.subtract(img, opened)
    _, mask = cv2.threshold(texture, thresh, 255, cv2.THRESH_BINARY)
    return texture, mask

# ——— Exemplo de uso ———

# Carregue sua imagem em escala de cinza:
img = cv2.imread('img/horse.png', cv2.IMREAD_GRAYSCALE)

# 4.1
dil = dilate(img, ksize=5)
ero = erode(img, ksize=5)

# 4.2
ab = opening(img, ksize=5)
fe = closing(img, ksize=5)

# 4.3
marker = erode(img, ksize=7)
rec_dil = reconstruction_by_dilation(marker, img)
marker2 = dilate(img, ksize=7)
rec_ero = reconstruction_by_erosion(marker2, img)

# 4.4
grad = morphological_gradient(img, ksize=5)

# 4.5
th = top_hat(img, ksize=15)
bh = bottom_hat(img, ksize=15)

# 4.6
rads, areas = granulometry(img, max_radius=15, step=2)

# 4.7
tex, seg = texture_segmentation(img, ksize=21, thresh=30)

# Exibição resumida
titles = [
    'Orig.', 'Dil.', 'Ero.', 'Ab.', 'Fe.',
    'Rec Dil.', 'Rec Ero.', 'Grad.', 'Top-Hat',
    'Bot-Hat', 'Granulo.', 'Text.', 'Seg.'
]
images = [
    img, dil, ero, ab, fe,
    rec_dil, rec_ero, grad, th,
    bh, # para granulometria vamos plotar gráfico
    tex, seg
]

plt.figure(figsize=(12, 8))
for i, im in enumerate(images[:-2]):
    plt.subplot(4, 4, i+1)
    plt.imshow(im, cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

# Granulometria (plot)
plt.subplot(4, 4, 11)
plt.plot(rads, areas, marker='o')
plt.title('Granulometria')
plt.xlabel('Raio')
plt.ylabel('Área')

# Textura e segmentação
plt.subplot(4, 4, 12)
plt.imshow(images[10], cmap='gray')
plt.title('Textura')
plt.axis('off')
plt.subplot(4, 4, 13)
plt.imshow(images[11], cmap='gray')
plt.title('Segmentação')
plt.axis('off')

plt.tight_layout()
plt.show()
