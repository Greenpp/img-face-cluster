import torch
from facenet_pytorch import InceptionResnetV1
from torch import Tensor


class Encoder:
    def __init__(self, gpu: bool = True) -> None:
        self.device = 'cuda' if gpu else 'cpu'
        self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

    def encode_faces(self, faces: list[Tensor]) -> Tensor:
        batch = torch.stack(faces).to(self.device)

        embeddings = self.model(batch).detach().cpu()
        return embeddings
