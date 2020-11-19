import glob
import pickle as pkl

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torch.nn.functional import l1_loss

import config

from .detection import get_faces_from_img
from .image import load_img
from .recognition import get_faces_embedding


def config_device() -> str:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device


def add_person(
    d: dict, face: Image.Image, embedding: torch.Tensor, f_name: str
) -> None:
    new_idx = len(d) + 1
    d[f'person_{new_idx}'] = {
        'face': np.array(face),
        'embedding': embedding,
        'imgs': set([f_name]),
    }


def save_people(people: dict) -> None:
    save_path = f'{config.BASE_DIR}/face_recognition_cache.pkl'
    with open(save_path, 'wb') as f:
        pkl.dump(people, f, pkl.HIGHEST_PROTOCOL)


# TODO describe everything

if __name__ == "__main__":
    device = config_device()
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    people = {}
    other = set()
    for extension in config.IMG_EXT:
        search_pattern = f'{config.BASE_DIR}/**/*.{extension}'
        base_len = len(config.BASE_DIR) + 1
        for f_name in glob.iglob(search_pattern, recursive=True):
            f_name = f_name[base_len:]
            img = load_img(f_name)

            faces = get_faces_from_img(img, mtcnn)
            if not faces:
                other.add(f_name)
            else:
                embeddings = get_faces_embedding(faces, resnet)

                for face, embedding in zip(faces, embeddings.unbind()):
                    new_idx = len(people) + 1
                    if not people:
                        add_person(people, face, embedding, f_name)
                    else:
                        found = False
                        for person in people.values():
                            distance = l1_loss(embedding, person['embedding']).item()
                            if distance < config.RECOGNITION_THRESHOLD:
                                found = True
                                person['imgs'].add(f_name)
                                break

                        if not found:
                            add_person(people, face, embedding, f_name)

    people['none'] = {'imgs': other}

    save_people(people)
