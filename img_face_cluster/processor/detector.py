import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from torch import Tensor


class Detector:
    def __init__(self, gpu: bool = True) -> None:
        device = 'cuda' if gpu else 'cpu'
        self.model = MTCNN(keep_all=True, device=device)

    def extract_faces(
        self,
        img: Image.Image,
    ) -> tuple[Tensor, np.ndarray, np.ndarray]:
        boxes, probs = self.model.detect(img)
        faces = self.model.extract(img, boxes, None)

        return faces, probs, boxes
