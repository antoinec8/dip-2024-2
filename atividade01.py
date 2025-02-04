import argparse
import numpy as np
import cv2

def generate_image(seed, width, height, mean, std):
    """
    Generates a grayscale image with pixel values sampled from a normal distribution.

    Args:
        seed (int): Random seed for reproducibility (student's registration number).
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        image (numpy.ndarray): The generated image.
    """
    ### START CODE HERE ###
    np.random.seed(seed)
    image = np.random.normal(mean, std, (height, width))
    image = np.clip(image, 0, 255).astype(np.uint8)
    ### END CODE HERE ###
    return image

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate an image with pixel values sampled from a normal distribution.")

    parser.add_argument('--registration_number', type=int, required=True, help="Student's registration number (used as seed)")
    parser.add_argument('--width', type=int, required=True, help="Width of the image")
    parser.add_argument('--height', type=int, required=True, help="Height of the image")
    parser.add_argument('--mean', type=float, required=True, help="Mean of the normal distribution")
    parser.add_argument('--std', type=float, required=True, help="Standard deviation of the normal distribution")
    parser.add_argument('--output', type=str, required=True, help="Path to save the generated image")

    args = parser.parse_args()

    # Generate the image
    image = generate_image(args.registration_number, args.width, args.height, args.mean, args.std)

    # Save the generated image
    cv2.imwrite(args.output, image)

    print(f"Image successfully generated and saved to {args.output}")

if __name__ == "__main__":
    main()


"""
    --registration_number: número da matrícula (19211015)
    --widht: largura da imagem (800)
    --height: altura da imagem (800)
    Isso irá gerar uma imagem de tamanho médio, que é processada rapidamente e contém boa visualização de detalhes.
    --mean: valores dos pixels (127)
    Em uma imagem em escala de cinza, os valores dos pixels variam de 0 (preto) a 255 (branco). Logo, a média, 127,
    resulta em uma imagem com tons de cinza intermediários, deixando o ruído com boa visualização.
    --std: desvio padrão (50)
    O desvio padrão controla a intensidade do ruído. Um valor de 50 significa que a maioria dos valores dos pixels 
    estará dentro de ±50 em torno da média. Isso cria um ruído evidente, mas ainda controlado.
    --output: nome da imagem que será gerada (output_atividade01)

    
    Antes de executar o programa, no terminal, adicione os seguintes comandos: 
    --registration_number 19211015 --width 800 --height 800 --mean 127 --std 50 --output output_atividade01.png
    Exemplo: python -u "c:\Local do arquivo"--registration_number 19211015 --width 800 --height 800 --mean 127 --std 50 --output output_atividade01.png
"""