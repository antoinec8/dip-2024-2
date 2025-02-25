import numpy as np

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """
    
    ### START CODE HERE ###
    height, width = image.shape[:2]
    depth = image.shape[2] if len(image.shape) == 3 else 1
    dtype = image.dtype
    min = np.min(image)
    max = np.max(image)
    mean = np.mean(image)
    std = np.std(image)
    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min,
        "max_value": max,
        "mean": mean,
        "std_dev": std
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")