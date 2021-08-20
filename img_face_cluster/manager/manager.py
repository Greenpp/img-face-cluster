import hashlib

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..processor import Detector, Encoder
from ..provider import Provider
from .models import Face, Photo
from .storage import Storage


class Manager:
    def __init__(self, debug: bool = False) -> None:
        self.storage = Storage(debug=debug)
        self.storage.init_database()

        self.detector = Detector()
        self.encoder = Encoder()

    def _create_provider(self, path: str, ext: list[str]) -> Provider:
        provider = Provider(path, ext)

        return provider

    def _calculate_hash(self, img: Image.Image) -> str:
        h_func = hashlib.sha256()

        h_func.update(img.tobytes())
        hashed_img = h_func.hexdigest()

        return hashed_img

    def _encode_array(self, arr: np.ndarray) -> bytes:
        return arr.tobytes()

    def scan(
        self,
        path: str,
        group: str,
        extensions: list[str] = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG'],
    ):
        provider = self._create_provider(path, extensions)

        # TODO add multiprocessing for hashing
        hashes = []
        paths = []
        for img, img_path in tqdm(provider, desc='Hashing'):
            img_hash = self._calculate_hash(img)

            hashes.append(img_hash)
            paths.append(img_path)

        # TODO add buffer for bigger encoding batches (multiple img at once)
        new_img_mask = self.storage.filter_new_images(hashes)
        for new, img_hash, img_path in tqdm(
            zip(new_img_mask, hashes, paths),
            desc='Extracting',
            total=sum(new_img_mask),
        ):
            if not new:
                continue
            img = Image.open(img_path)
            faces, probs, boxes = self.detector.extract_faces(img)
            embeddings = self.encoder.encode_faces(faces)

            img_dict = dict(
                path=str(img_path),
                hash=img_hash,
            )
            faces_dicts = []
            if faces is not None:
                for embedding, prob, box in zip(embeddings, probs, boxes):
                    face_dict = dict(
                        probability=prob,
                        bbox=self._encode_array(box),
                        encoding=self._encode_array(embedding),
                    )
                    faces_dicts.append(face_dict)

            self.storage.save_image(img_dict, faces_dicts, group)

    def cluster(self, group: str):
        # Get all faces from group
        # Cluster
        # Assign tags
        pass
