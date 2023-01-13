import albumentations as A

p = 0.3
default_augment = A.Compose([
    # Changing image coloring
    A.OneOf([
        A.CLAHE(p=p),
        A.ColorJitter(p=p),
        A.Emboss(p=p),
        A.Equalize(p=p),
        A.FancyPCA(p=p),
        A.HueSaturationValue(p=p),
        A.InvertImg(p=p),
        A.RandomBrightnessContrast(p=p),
        A.RGBShift(p=p),
    ]),

    # Grayscale
    A.OneOf([
        A.ToSepia(p=p),
        A.ToGray(p=p),
    ]),

    # Overlays
    # Disabled due to
    # https://github.com/albumentations-team/albumentations/issues/361
    # A.OneOf([
    #     A.RandomShadow(p=p),
    #     A.RandomFog(p=p),
    #     A.RandomSnow(p=p),
    #     A.RandomSunFlare(p=p),
    # ]),

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
        A.JpegCompression(p=p),
        A.GaussianBlur(p=p),
        A.Defocus(p=p),
        A.Downscale(p=p),
        A.Posterize(p=p),
        A.GlassBlur(p=p),
        A.MedianBlur(p=p),
        A.MotionBlur(p=p),
        A.ZoomBlur(p=p),
    ]),

    # Spatial transform
    A.OneOf([
        A.ElasticTransform(p=p),
        A.Perspective(p=p),
        A.PiecewiseAffine(p=p),
        A.ShiftScaleRotate(p=p),
        A.SafeRotate((-10, 10), p=p),
    ])
])
