from torchvision import transforms
import numpy as np
from pil import Image
import colorsys

class ColorizeTransform(object):
    def __init__(self, intensity_range=(-50, 50)):
      self.intensity_range = intensity_range


    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img,dtype=np.float32)

        # Randomly adjust RGB values
        intensity_shift = np.random.uniform(self.intensity_range[0], self.intensity_range[1],size=3)

        # Apply the intensity shift to each channel independently
        img_np += intensity_shift

        # Clip values to the valid range [0, 255]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Convert numpy array back to PIL Image
        result_img = Image.fromarray(img_np)

        return result_img