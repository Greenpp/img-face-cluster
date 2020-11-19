import glob
import pickle as pkl
from typing import Iterable

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from detection import get_faces_from_img
from image import load_img
from recognition import get_faces_embedding

# TODO mean embedding with all found faces ?


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
        self.people = {'none': {'paths': set()}}

    def _save_results(
        self,
        dir: str,
    ) -> None:
        save_path = f'{dir}/face-recognition-results.pkl'
        with open(save_path, 'wb') as f:
            pkl.dump(self.people, f, pkl.HIGHEST_PROTOCOL)

    def _add_new_person(
        self,
        base_face: Image.Image,
        embedding: torch.Tensor,
        f_name: str,
    ) -> None:
        new_idx = len(self.people)
        key = f'person_{new_idx}'

        self.people[key] = {
            'face': np.array(base_face),
            'embedding': embedding,
            'paths': set([f_name]),
        }

    def _add_img_without_people(self, f_name: str) -> None:
        self.people['none']['paths'].add(f_name)

    def scan(
        self,
        search_dir: str,
        extensions: Iterable[str],
        detection_threshold: float = 0.95,
        recognition_threshold: float = 20,
        img_resize_width: int = 1280,
        img_resize_height: int = 720,
        verbose: bool = True,
    ) -> None:
        self._reset_cache()

        base_path_len = len(search_dir)
        for extension in extensions:
            search_pattern = f'{search_dir}/**/*.{extension}'
            for f_path in glob.iglob(search_pattern, recursive=True):
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
                    new_faces = 0

                    for face, embedding in zip(faces, embeddings.unbind()):
                        found = False
                        for person, person_dict in self.people.items():
                            if person != 'none':
                                base_embedding = person_dict['embedding']
                                distance = torch.dist(
                                    embedding, base_embedding, p=1
                                ).item()
                                if distance < recognition_threshold:
                                    found = True
                                    person_dict['paths'].add(rel_path)
                                    break

                        if not found:
                            new_faces += 1
                            self._add_new_person(face, embedding, rel_path)

                    if verbose:
                        print(f'found {len(faces)} face(s) ({new_faces} new)')

        self._save_results(search_dir)
