from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def _get_img_orientation(img: Image.Image) -> int:
    ORIENTATION_TAG = 274
    EXIF_ORIENTATION = {
        1: 0,
        2: 0,
        3: 180,
        4: 180,
        5: 90,
        6: 90,
        7: -90,
        8: -90,
    }

    metadata = img.getexif()
    orientation = 0
    if ORIENTATION_TAG in metadata:
        orientation_code = metadata[ORIENTATION_TAG]
        if orientation_code in EXIF_ORIENTATION:
            orientation = EXIF_ORIENTATION[orientation_code]

    return orientation


def flip_img(img: Image.Image) -> Image.Image:
    orientation = _get_img_orientation(img)

    if orientation != 0:
        img = img.rotate(-orientation, expand=True)

    return img


def resize_img(
    img: Image.Image,
    max_width: int,
    max_height: int,
) -> Image.Image:
    resize_factor = max(
        max_width / img.width,
        max_height / img.height,
    )
    new_width = int(img.width * resize_factor)
    new_height = int(img.height * resize_factor)

    img = img.resize((new_width, new_height))

    return img
