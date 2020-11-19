from typing import Tuple

from PIL import Image


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


def get_new_img_size(
    img: Image.Image,
    max_width: int,
    max_height: int,
) -> Tuple[int, int]:
    # Resize factor to keep ratio
    resize_factor = max(
        max_width / img.width,
        max_height / img.height,
    )

    new_width = int(img.width * resize_factor)
    new_height = int(img.height * resize_factor)

    return new_width, new_height


def resize_img(
    img: Image.Image,
    max_width: int,
    max_height: int,
) -> Image.Image:
    if img.width > max_width or img.height > max_height:
        new_size = get_new_img_size(img, max_width, max_height)
        img = img.resize(new_size)

    return img


def flip_img(img: Image.Image) -> Image.Image:
    flip = get_img_flip(img)

    if flip != 0:
        img = img.rotate(flip, expand=True)

    return img


def load_img(
    path: str,
    resize_max_width: int,
    resize_max_height: int,
) -> Image.Image:
    img = Image.open(path)
    img = resize_img(img, resize_max_width, resize_max_height)
    img = flip_img(img)

    return img
