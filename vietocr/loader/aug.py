import albumentations as A

p = 0.3
default_augment = A.Compose([
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
