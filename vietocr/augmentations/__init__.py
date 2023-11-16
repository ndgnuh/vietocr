import random

import albumentations as A

from .bloom_filter import BloomFilter
from .chromatic_abberation import ChromaticAberration
from .custom_augmentations import ColorPatchOverlay, ScaleDegrade
from .fake_light import FakeLight
from .fbm_noise import RandomFbmNoise


class RandomOrderCompose(A.Compose):
    def __call__(self, *a, **kw):
        random.shuffle(self.transforms)
        return super().__call__(*a, **kw)


def dummy_transform(image):
    return dict(image=image)


def get_augmentation(p: float = 0.5):
    # +--------------------------------------------------------------+
    # | If probability is less than 0, return a dummy transformation |
    # +--------------------------------------------------------------+
    if p <= 0:
        return dummy_transform

    default_augment = RandomOrderCompose(
        [
            # +----------------------------------------+
            # | Overlay with some kind of color filter |
            # +----------------------------------------+
            A.OneOf(
                [
                    ColorPatchOverlay(),
                    FakeLight(),
                    BloomFilter(),
                    ChromaticAberration(),
                ],
                p=p,
            ),
            # +-------------------------+
            # | Changing image coloring |
            # +-------------------------+
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
            # +------------------------------------------------------------------+
            # | Road overlays, weird                                             |
            # | Fog, snow, sunflare are disabled                                 |
            # | due to deadlock bug and readability                              |
            # | https://github.com/albumentations-team/albumentations/issues/361 |
            # +------------------------------------------------------------------+
            A.RandomShadow(p=p),
            # +---------------+
            # | Random noises |
            # +---------------+
            A.OneOf(
                [
                    RandomFbmNoise(),
                    A.ISONoise(),
                    A.MultiplicativeNoise(),
                ],
                p=p,
            ),
            # +-----------------+
            # | Random dropouts |
            # +-----------------+
            A.OneOf(
                [
                    A.PixelDropout(),
                    A.ChannelDropout(),
                ],
                p=p,
            ),
            # +------------------------+
            # | Random image degration |
            # +------------------------+
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
            # +-------------------------+
            # | Spatial transformations |
            # +-------------------------+
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

    # +------------------------------+
    # | Return composed augmentation |
    # +------------------------------+
    return default_augment
