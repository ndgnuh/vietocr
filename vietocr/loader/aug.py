import albumentations as A

p = 0.3
default_augment = A.Compose([
    A.CLAHE(p=p),
    A.ColorJitter(p=p)
])
