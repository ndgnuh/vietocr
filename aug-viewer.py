import streamlit as st
import albumentations as A
import inspect
import numpy as np
from vietocr.loader.aug import default_augment
from PIL import Image

all_augmentations = [
    "all",
    "SafeRotate",
    "ShiftScaleRotate",
    "RandomShadow",
    "RandomFog",
    "RandomSnow",
    "RandomSunFlare",
    "CLAHE",
    "ColorJitter",
    "Emboss",
    "Equalize",
    "FancyPCA",
    "HueSaturationValue",
    "InvertImg",
    "RandomBrightnessContrast",
    "RGBShift",
    "ToSepia",
    "ToGray",
    "ISONoise",
    "MultiplicativeNoise",
    "PixelDropout",
    "ChannelDropout",
    "ImageCompression",
    "GaussianBlur",
    "Defocus",
    "Posterize",
    "GlassBlur",
    "MedianBlur",
    "MotionBlur",
    "ZoomBlur",
    "ElasticTransform",
    "Perspective",
    "PiecewiseAffine",
]


image = st.file_uploader("Upload image")

if image is None:
    st.stop()

augmentation = st.selectbox("Augmentation", all_augmentations)
if augmentation == "all":
    def Aug(p=None, always_apply=None): return default_augment
else:
    Aug = getattr(A, augmentation)
cols = st.columns(4)
sigs = inspect.signature(Aug)
st.write(dict(sigs.parameters))
kwargs = dict(p=1.0, always_apply=True)
i = 0
for name, param in sigs.parameters.items():
    if name in ["always_apply", "p"]:
        continue
    i = (i + 1) % len(cols)
    try:
        with cols[i]:
            value = st.text_input(name).strip()
            if len(value) > 0:
                kwargs[name] = eval(value)
    except Exception:
        st.warning(f"No {name}")

rerun = st.button("Rerun")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Original")
    st.image(image)
with c2:
    st.subheader("Augmented")
    aug = Aug(**kwargs)
    image = np.array(Image.open(image).convert("RGB"))
    if rerun:
        st.image(aug(image=image)['image'])
