import numpy as np 
from PIL import ImageEnhance, Image

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, brightness = 0, contrast = 0, sharpness = 0, color = 0):
        transform_dict = {"brightness":brightness, "contrast":contrast, "sharpness":sharpness, "color":color}
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
    
    def __call__(self, img):
        out = img
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
    
        return out