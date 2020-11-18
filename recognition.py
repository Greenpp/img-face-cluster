from typing import TYPE_CHECKING, Iterable

import torch
from PIL import Image

from .image import transform_img_to_tensor

if TYPE_CHECKING:
    from facenet_pytorch import InceptionResnetV1


def get_faces_embedding(
    faces: Iterable[Image.Image],
    resnet: InceptionResnetV1,
) -> torch.Tensor:
    t_faces = []
    for face in faces:
        t_face = transform_img_to_tensor(face)
        t_faces.append(t_face)

    face_batch = torch.stack(t_faces)
    embeddings = resnet(face_batch)

    return embeddings
