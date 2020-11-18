from typing import TYPE_CHECKING, Iterable

import torch
from PIL import Image
from torchvision import transforms as T

if TYPE_CHECKING:
    from facenet_pytorch import InceptionResnetV1


def transform_img_to_tensor(img: Image.Image) -> torch.Tensor:
    # Normalize using standard Image Net values
    transform = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    t_img = transform(img).unsqueeze(0)

    return t_img


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
