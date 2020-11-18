from typing import TYPE_CHECKING, Iterable

from PIL import Image

if TYPE_CHECKING:
    from facenet_pytorch import MTCNN


def get_faces_from_img(
    img: Image.Image,
    mtcnn: MTCNN,
) -> Iterable[Image.Image]:
    boxes, _ = mtcnn.detect(img)

    faces = []
    for box in boxes:
        f_img = img.crop(box).resize((160, 160))
        faces.append(f_img)

    return faces
