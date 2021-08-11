import logging
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

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


class Provider(Dataset):
    def __init__(self, img_root: str, img_ext: list[str], max_size: int = 1280) -> None:
        self.max_size = max_size

        all_img_paths = []
        for ext in img_ext:
            paths = self._get_img_paths(img_root, ext)
            all_img_paths.extend(paths)

        self.img_paths = all_img_paths

    def _get_img_paths(self, root: str, ext: str) -> list[Path]:
        paths = Path(root).glob(f'**/*.{ext}')

        return list(paths)

    def __getitem__(self, index) -> Image.Image:
        img_path = self.img_paths[index]

        img = Image.open(img_path)

        # Preprocessing
        img = self._resize_img(img)
        img = self._reset_img_orientation(img)

        return img

    def _resize_img(self, img: Image.Image) -> Image.Image:
        logging.debug(f'Resizing image with size {img.width}x{img.height}')
        bigger_side = max(img.height, img.width)
        if bigger_side <= 1280:
            logging.debug('Image is smaller than max size, skipping')
            return img

        resize_ratio = bigger_side / self.max_size
        new_height = int(img.height / resize_ratio)
        new_width = int(img.width / resize_ratio)

        resized_img = img.resize((new_width, new_height))
        logging.debug(f'Image resized to {new_width}x{new_height}')
        return resized_img

    def _reset_img_orientation(self, img: Image.Image) -> Image.Image:
        logging.debug('Resetting image orientation')

        mirrored = False
        rotation = 0
        try:
            metadata = img.getexif()
            orientation_code = metadata[ORIENTATION_TAG]

            rotation = EXIF_ORIENTATION[orientation_code]
            mirrored = not bool(orientation_code % 2)
            logging.debug(
                f'Rotation metadata parsed, rotation: {rotation} | mirrored: {mirrored}'
            )
        except KeyError:
            logging.debug('Could not read rotation metadata')

        if rotation != 0:
            img = img.rotate(-rotation, expand=True)
        if mirrored:
            img = ImageOps.mirror(img)

        return img

    def __len__(self) -> int:
        return len(self.img_paths)
