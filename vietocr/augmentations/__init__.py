import random

import albumentations as A

from .custom_augmentations import (
    ColorPatchOverlay,
    ScaleDegrade,
)
from .fbm_noise import RandomFbmNoise
from .chromatic_abberation import ChromaticAberration
from .bloom_filter import BloomFilter
from .fake_light import FakeLight


class RandomOrderCompose(A.Compose):
    def __call__(self, *a, **kw):
        random.shuffle(self.transforms)
        return super().__call__(*a, **kw)


def get_augmentation(p: float = 0.5):
    if p < 0:
        return None

    default_augment = RandomOrderCompose(
        [
            # Overlay
            A.OneOf(
                [
                    ColorPatchOverlay(),
                    FakeLight(),
                    BloomFilter(),
                    ChromaticAberration(),
                    # PatternOverlay(patterns="vietocr/data/patterns", p=1),
                ],
                p=p,
            ),
            # Changing image coloring
            A.OneOf(
                [
                    A.CLAHE(),
                    A.ColorJitter(),
                    A.Emboss(),
                    A.HueSaturationValue(),
                    A.RandomBrightnessContrast(),
                    A.InvertImg(),
                    A.RGBShift(),
                    A.ToSepia(),
                    A.ToGray(),
                ],
                p=p,
            ),
            # Overlays
            # Fog, snow, sunflare are disabled
            # due to deadlock bug and readability
            # https://github.com/albumentations-team/albumentations/issues/361
            A.RandomShadow(p=p),
            # Noises
            A.OneOf(
                [
                    RandomFbmNoise(),
                    A.ISONoise(),
                    A.MultiplicativeNoise(),
                ],
                p=p,
            ),
            # Dropouts
            A.OneOf(
                [
                    A.PixelDropout(),
                    A.ChannelDropout(),
                ],
                p=p,
            ),
            # Image degration
            A.OneOf(
                [
                    A.ImageCompression(),
                    ScaleDegrade(),
                    A.GaussianBlur(),
                    A.Defocus(),
                    A.Posterize(),
                    A.GlassBlur(sigma=0.1, max_delta=1, iterations=1),
                    A.MedianBlur(blur_limit=3),
                    A.MotionBlur(),
                    A.ZoomBlur(max_factor=1.1),
                ],
                p=p,
            ),
            # Spatial transform
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=1, alpha_affine=1),
                    A.Perspective(),
                    A.Affine(rotate=(-10, 10), fit_output=True)
                    # 4/2 is divisable by the image size
                    # in hope that it will not cause any crash
                    # Still crash
                    # A.PiecewiseAffine(nb_rows=4, nb_cols=2, always_apply=True),
                    # Removed due to making the output out of range
                    # A.ShiftScaleRotate(always_apply=True),
                    # A.SafeRotate((-10, 10), always_apply=True),
                ],
                p=p,
            ),
        ]
    )
    return default_augment
