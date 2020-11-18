import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from .detection import get_faces_from_img
from .image import load_img
from .recognition import get_faces_embedding


def config_device() -> str:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return device


if __name__ == "__main__":
    device = config_device()
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
