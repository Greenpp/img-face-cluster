from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


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

    def __getitem__(self, index) -> tuple[Image.Image, Path]:
        img_path = self.img_paths[index]

        img = Image.open(img_path)

        return img, img_path

    def __len__(self) -> int:
        return len(self.img_paths)
