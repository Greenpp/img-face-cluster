from typing import TYPE_CHECKING, Iterable
import config

from PIL import Image

if TYPE_CHECKING:
    from facenet_pytorch import MTCNN


def get_faces_from_img(
    img: Image.Image,
    mtcnn: 'MTCNN',
    threshold: float,
) -> Iterable[Image.Image]:
    boxes, probs = mtcnn.detect(img)

    faces = []
    filtered_probs = []
    for box, prob in zip(boxes, probs):
        if prob >= threshold:
            f_img = img.crop(box).resize((160, 160))
            faces.append(f_img)
            filtered_probs.append(prob)

    return faces, filtered_probs
