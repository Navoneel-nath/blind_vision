import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from madnet_keras import MADNet

def load_and_preprocess_image(image_path, width, height):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def display_disparity(disparity):
    plt.figure(figsize=(8, 6))
    plt.imshow(disparity, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title("Predicted Disparity Map")
    plt.axis("off")
    plt.show()

def main():
    WIDTH, HEIGHT = 640, 480
    left_image_path = r"C:\Users\navon\Documents\capstone\dataset\left\left.png"
    right_image_path = r"C:\Users\navon\Documents\capstone\dataset\right\right.png"
    left_img = load_and_preprocess_image(left_image_path, WIDTH, HEIGHT)
    right_img = load_and_preprocess_image(right_image_path, WIDTH, HEIGHT)
    left_input = np.expand_dims(left_img, axis=0)
    right_input = np.expand_dims(right_img, axis=0)
    model = MADNet()  
    weights_path = r"C:\Users\navon\Documents\capstone\madnet_keras\synthetic.h5"
    model.load_weights(weights_path)
    disparity = model.predict([left_input, right_input])
    disparity_map = np.squeeze(disparity, axis=0)
    if disparity_map.ndim == 3 and disparity_map.shape[2] == 1:
        disparity_map = disparity_map[:, :, 0]
    display_disparity(disparity_map)

if __name__ == '__main__':
    main()
