import albumentations as A
from os import path, listdir
import random
import numpy as np
import cv2


def random_crop(pattern, width, height):
    ph, pw = pattern.shape[:2]
    x0 = random.randint(0, pw - width)
    y0 = random.randint(0, ph - height)
    x1 = x0 + width
    y1 = y0 + height
    return pattern[y0:y1, x0:x1]


class PatternOverlay:
    def __init__(
        self,
        patterns: str,
        alphas=(0.4, 0.7),
        p: float = 0.5,
    ):
        if path.isdir(patterns):
            self.patterns = [cv2.imread(path.join(patterns, pattern))
                             for pattern in listdir(patterns)]
        elif path.isfile(patterns):
            self.patterns = [cv2.imread(patterns)]
        self.alphas = alphas
        self.p = p

    def __call__(self, image: np.ndarray, **kw):
        # **kw because of albumentation interface
        if random.random() > self.p:
            return dict(image=image, **kw)
        alpha = random.uniform(*self.alphas)
        pattern = random.choice(self.patterns)
        h, w = image.shape[:2]
        try:
            pattern = random_crop(pattern, width=w, height=h)
        except ValueError:
            pattern = cv2.resize(pattern, (w, h))
        image = image * alpha + pattern * (1 - alpha)
        image = np.clip(image, 0, 255).round().astype('uint8')
        return dict(image=image, **kw)


p = 0.3
default_augment = A.Compose([
    PatternOverlay(patterns="vietocr/data/patterns", p=p * 2),
    # Changing image coloring
    A.OneOf([
        A.CLAHE(p=p),
        A.ColorJitter(p=p),
        A.Emboss(p=p),
        A.HueSaturationValue(p=p),
        A.RandomBrightnessContrast(p=p),
        A.InvertImg(p=p),
        A.RGBShift(p=p),
        A.ToSepia(p=p),
        A.ToGray(p=p),
    ]),

    # Overlays
    # Fog, snow, sunflare are disabled
    # due to deadlock bug and readability
    # https://github.com/albumentations-team/albumentations/issues/361
    A.RandomShadow(p=p),

    # Noises
    A.OneOf([
        A.ISONoise(p=p),
        A.MultiplicativeNoise(p=p),
    ]),

    # Dropouts
    A.OneOf([
        A.PixelDropout(p=p),
        A.ChannelDropout(p=p),
    ]),

    # Image degration
    A.OneOf([
        A.ImageCompression(p=p),
        A.GaussianBlur(p=p),
        A.Defocus(radius=(1, 3), p=p),
        A.Posterize(p=p),
        A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, p=p),
        A.MedianBlur(blur_limit=3, p=p),
        A.MotionBlur(p=p),
        A.ZoomBlur(max_factor=1.1, p=p),
    ]),

    # Spatial transform
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=1, alpha_affine=1, p=p),
        A.Perspective(fit_output=True, p=p),

        # Removed due to bug
        # A.PiecewiseAffine(nb_rows=3, nb_cols=3, p=p),

        # Removed due to making the output out of range
        # A.ShiftScaleRotate(p=p),
        # A.SafeRotate((-10, 10), p=p),
    ])
])
