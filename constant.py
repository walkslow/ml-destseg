ALL_CATEGORY = [
    "capsule",
    "metal_nut",
    "pill",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "cable",
    "bottle",
    "grid",
    "hazelnut",
    "leather",
    "tile",
    "carpet",
    "screw",
]
RESIZE_SHAPE = [256, 256]  # width * height
# For 3-channel RGB images, using ImageNet standards
NORMALIZE_MEAN_RGB = [0.485, 0.456, 0.406]
NORMALIZE_STD_RGB = [0.229, 0.224, 0.225]

# For 1-channel grayscale images
NORMALIZE_MEAN_L = [0.5]
NORMALIZE_STD_L = [0.5]
