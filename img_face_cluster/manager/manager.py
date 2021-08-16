from pathlib import Path

from ..processor import Detector, Encoder
from ..provider import Provider
from .cache import Cache


class Manager:
    def __init__(self) -> None:
        self.cache = Cache()

        self.detector = Detector()
        self.encoder = Encoder()
        self.provider = None

    def create_provider(self, path: str, ext: list[str]) -> None:
        self.provider = Provider(path, ext)
