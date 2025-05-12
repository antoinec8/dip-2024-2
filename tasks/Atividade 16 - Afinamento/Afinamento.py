import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem binária
img = cv2.imread('img/horse.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

# Elementos estruturantes para Hit-or-Miss afinamento
# Conjunto com 8 pares de padrões (representando rotações)
def get_structuring_elements():
    elems = []
    # Definimos os pares J e K como (J, K)
    patterns = [
        ([ [0, 0, 0],
           [-1, 1, -1],
           [1, 1, 1] ],
         [ [-1, -1, -1],
           [ 0,  0,  0],
           [-1, -1, -1] ]),

        ([ [-1, 0, 0],
           [1, 1, 0],
           [-1, 1, -1] ],
         [ [ 0, -1, -1],
           [-1, 0, -1],
           [ 0, -1,  0] ]),

        ([ [1, -1, 0],
           [1, 1, 0],
           [1, -1, 0] ],
         [ [-1, -1, -1],
           [ 0,  0, -1],
           [-1, -1, -1] ]),

        ([ [-1, 1, -1],
           [0, 1, 1],
           [0, 0, -1] ],
         [ [ 0, -1, 0],
           [-1, 0, -1],
           [-1, -1, 0] ])
    ]
    
    # Gera rotações horizontais dos padrões
    for J, K in patterns:
        for i in range(4):
            J_rot = np.rot90(J, i)
            K_rot = np.rot90(K, i)
            elems.append((np.array(J_rot, dtype=np.int8), np.array(K_rot, dtype=np.int8)))
    return elems

# Função de afinamento por Hit or Miss
def thinning_hit_or_miss(img_bin):
    img_bin = img_bin.copy()
    elems = get_structuring_elements()
    changed = True

    while changed:
        changed = False
        for J, K in elems:
            hit = cv2.morphologyEx(img_bin, cv2.MORPH_HITMISS, J)
            if np.any(hit):
                img_bin[hit == 1] = 0
                changed = True
    return img_bin * 255 

# Aplicar afinamento
thin = thinning_hit_or_miss(binary)

# Exibir resultado
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(binary * 255, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thin, cmap='gray')
plt.title("Afinamento usando Hit or Miss")
plt.axis('off')

plt.tight_layout()
plt.show()