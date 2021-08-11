from facenet_pytorch import MTCNN
from PIL import Image
from torch import Tensor


class Detector:
    def __init__(self, face_threshold: float = 0.0, gpu: bool = True) -> None:
        device = 'cuda' if gpu else 'cpu'
        self.model = MTCNN(keep_all=True, device=device)

        self.face_threshold = face_threshold

    def extract_faces(self, img: Image.Image) -> list[Tensor]:
        faces, probs = self.model(img, return_prob=True)

        filtered_faces = []
        for face, prob in zip(faces, probs):
            if prob > self.face_threshold:
                filtered_faces.append(face)

        return filtered_faces
