from typing import Tuple

from PIL import Image

import config


def get_img_flip(img: Image.Image) -> int:
    ORIENTATION_TAG = 274
    EXIF_ORIENTATION_FLIP = {
        1: 0,
        2: 0,
        3: 180,
        4: 180,
        5: -90,
        6: -90,
        7: 90,
        8: 90,
    }

    img_orientation = img.getexif()[ORIENTATION_TAG]
    if img_orientation in EXIF_ORIENTATION_FLIP:
        flip = EXIF_ORIENTATION_FLIP[img_orientation]
    else:
        flip = 0

    return flip


def get_new_img_size(img: Image.Image) -> Tuple[int, int]:
    # Resize factor to keep ratio
    resize_factor = max(
        config.MAX_IMG_WIDTH / img.width,
        config.MAX_IMG_HEIGHT / img.height,
    )

    new_width = int(img.width * resize_factor)
    new_height = int(img.height * resize_factor)

    return new_width, new_height


def resize_img(img: Image.Image) -> Image.Image:
    if img.width > config.MAX_IMG_WIDTH or img.height > config.MAX_IMG_HEIGHT:
        new_size = get_new_img_size(img)
        img = img.resize(new_size)

    return img


def flip_img(img: Image.Image) -> Image.Image:
    flip = get_img_flip(img)

    if flip != 0:
        img = img.rotate(flip, expand=True)

    return img


def load_img(base_dir_path: str) -> Image.Image:
    img_path = f'{config.BASE_DIR}/{base_dir_path}'

    img = Image.open(img_path)
    img = resize_img(img)
    img = flip_img(img)

    return img
