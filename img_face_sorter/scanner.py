import glob
import pickle as pkl
from typing import Iterable

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from .detection import get_faces_from_img
from .image import load_img
from .recognition import get_faces_embedding


class Scanner:
    def __init__(self) -> None:
        self._config_device()

        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.face_encoder = InceptionResnetV1(pretrained='vggface2').eval()

        self._reset_cache()

    def _config_device(self) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.device = device

    def _reset_cache(self) -> None:
        self.cache = {
            'empty': [],
            'people': [],
        }

    def _save_cache(
        self,
        dir: str,
        idx: int,
    ) -> None:
        save_path = f'{dir}/scan-cache-{idx}.pkl'
        with open(save_path, 'wb') as f:
            pkl.dump(self.cache, f, pkl.HIGHEST_PROTOCOL)

    def _add_new_face(
        self,
        base_face: Image.Image,
        embedding: torch.Tensor,
        f_name: str,
    ) -> None:
        self.cache['people'].append(
            {
                'face': np.array(base_face),
                'embedding': embedding,
                'path': f_name,
            }
        )

    def _add_img_without_people(self, f_name: str) -> None:
        self.cache['empty'].append(f_name)

    def scan(
        self,
        search_dir: str,
        extensions: Iterable[str],
        detection_threshold: float = 0.95,
        img_resize_width: int = 1280,
        img_resize_height: int = 720,
        verbose: bool = True,
        save_every: int = 100,
    ) -> None:
        self._reset_cache()

        base_path_len = len(search_dir)
        file_num = 0
        for extension in extensions:
            search_pattern = f'{search_dir}/**/*.{extension}'
            for f_path in glob.iglob(search_pattern, recursive=True):
                file_num += 1

                # Leave only relative path to search directory
                rel_path = f_path[base_path_len:]

                if verbose:
                    print(f'Scanning {rel_path} ... ', end='', flush=True)

                img = load_img(f_path, img_resize_width, img_resize_height)
                faces = get_faces_from_img(img, self.face_detector, detection_threshold)

                if not faces:
                    self._add_img_without_people(rel_path)

                    if verbose:
                        print('no faces found')
                else:
                    embeddings = get_faces_embedding(faces, self.face_encoder)

                    for face, embedding in zip(faces, embeddings.unbind()):
                        self._add_new_face(face, embedding, rel_path)

                    if verbose:
                        print(f'found {len(faces)} face(s)')

                if file_num % save_every == 0:
                    idx = file_num // save_every
                    self._save_cache(search_dir, idx)
                    self._reset_cache()

        idx = (file_num // save_every) + 1
        self._save_cache(search_dir, idx)
