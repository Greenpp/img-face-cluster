from typing import Union
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from torch import Tensor


class Detector:
    def __init__(self, gpu: bool = True) -> None:
        device = 'cuda' if gpu else 'cpu'
        self.model = MTCNN(keep_all=True, device=device, min_face_size=35)

    def extract_faces(
        self,
        img: Union[Image.Image, list[Image.Image]],
    ) -> tuple[Tensor, np.ndarray, np.ndarray]:
        boxes, probs = self.model.detect(img)
        faces = self.model.extract(img, boxes, None)

        return faces, probs, boxes
