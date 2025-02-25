import argparse
import numpy as np
import cv2 as cv
import requests

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    image_url = requests.get(url)
    if image_url.status_code != 200:
        raise Exception(f"Não foi possível baixar a imagem. Erro: {image_url.status_code}.")
    image_np = np.frombuffer(image_url.content, dtype=np.uint8)
    image = cv.imdecode(image_np, **kwargs)
    ### END CODE HERE ###
    
    return image

if __name__ == "__main__":
    url = "https://cdn-images.dzcdn.net/images/cover/7ce6b8452fae425557067db6e6a1cad5/0x1900-000000-80-0-0.jpg"
    image = load_image_from_url(url, flags=cv.IMREAD_UNCHANGED)
    if image is not None:
        image_resized = cv.resize(image, (600, 400))
        cv.imshow("imagem (redimensionada)", image_resized)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Não foi possível carregar a imagem escolhida.")