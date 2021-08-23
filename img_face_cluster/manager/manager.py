import hashlib

import numpy as np
from PIL import Image
from sklearn.cluster import AffinityPropagation
from sqlalchemy.orm import session
from tqdm import tqdm

from ..processor import Detector, Encoder
from ..provider import Provider
from .models import Cluster, Face, Group, Photo
from .storage import Storage


class Manager:
    def __init__(self, debug: bool = False, gpu: bool = True) -> None:
        self.storage = Storage(debug=debug)
        self.storage.init_database()

        self.detector = Detector(gpu=gpu)
        self.encoder = Encoder(gpu=gpu)

    def _create_provider(self, path: str, ext: list[str]) -> Provider:
        provider = Provider(path, ext)

        return provider

    def _calculate_hash(self, img: Image.Image) -> str:
        h_func = hashlib.sha256()

        h_func.update(img.tobytes())
        hashed_img = h_func.hexdigest()

        return hashed_img

    def _encode_array(self, arr: np.ndarray) -> bytes:
        return arr.tobytes()

    def _decode_array(self, b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32)

    def scan(
        self,
        path: str,
        group: str,
        extensions: list[str] = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG'],
    ):
        provider = self._create_provider(path, extensions)

        # TODO add multiprocessing for hashing
        hashes = []
        paths = []
        for img, img_path in tqdm(provider, desc='Hashing'):
            img_hash = self._calculate_hash(img)

            hashes.append(img_hash)
            paths.append(img_path)

        # TODO add buffer for bigger encoding batches (multiple img at once)
        new_img_mask = self.storage.filter_new_images(hashes)
        new_img_count = sum(new_img_mask)
        print(f'Found {new_img_count} new images')
        for new, img_hash, img_path in tqdm(
            zip(new_img_mask, hashes, paths),
            desc='Extracting',
            total=new_img_count,
        ):
            if not new:
                continue
            img = Image.open(img_path)
            faces, probs, boxes = self.detector.extract_faces(img)
            embeddings = self.encoder.encode_faces(faces)

            img_dict = dict(
                path=str(img_path),
                hash=img_hash,
            )
            faces_dicts = []
            if faces is not None:
                for embedding, prob, box in zip(embeddings, probs, boxes):
                    face_dict = dict(
                        probability=prob,
                        bbox=self._encode_array(box),
                        encoding=self._encode_array(embedding),
                    )
                    faces_dicts.append(face_dict)

            self.storage.save_image(img_dict, faces_dicts, group)

    def cluster(self, group: str):
        # TODO add probability threshold
        # TODO when more than one face from single photo is in the same cluster, leave most probable one ?
        session = self.storage.get_session()
        group_orm = session.query(Group).filter(Group.name == group).first()
        group_photos = session.query(Photo).filter(Photo.group_id == group_orm.id).all()
        group_photo_ids = [p.id for p in group_photos]

        group_faces = (
            session.query(Face).filter(Face.photo_id.in_(group_photo_ids)).all()
        )

        embedidngs = [self._decode_array(f.encoding) for f in group_faces]
        cluster_alg = AffinityPropagation(random_state=42)

        cluster_alg.fit(embedidngs)
        clusters = cluster_alg.labels_
        center_ids = cluster_alg.cluster_centers_indices_

        clusters_orm = []
        for i, idx in enumerate(center_ids):
            center_face = group_faces[idx]
            new_cluster = Cluster(name=str(i), center=center_face.id)
            clusters_orm.append(new_cluster)
        session.add_all(clusters_orm)
        session.commit()

        for face, c in zip(group_faces, clusters):
            face.cluster_id = clusters_orm[c].id
        session.commit()

        session.close()

    def _resize_img(self, img: Image.Image, max_size: int = 128) -> Image.Image:
        bigger_side = max(img.width, img.height)
        resize_factor = bigger_side / max_size

        new_width = int(img.width / resize_factor)
        new_height = int(img.height / resize_factor)

        return img.resize((new_width, new_height), Image.BICUBIC)

    def show_cluster(self, cluster: str) -> None:
        session = self.storage.get_session()

        cluster_orm = session.query(Cluster).filter(Cluster.name == cluster).first()
        face_photo_join = (
            session.query(Face, Photo)
            .filter(Face.cluster_id == cluster_orm.id)
            .join(Photo)
            .all()
        )

        faces = []
        for face, photo in face_photo_join:
            img = Image.open(photo.path)
            bbox = self._decode_array(face.bbox)

            face = img.crop(bbox)
            faces.append(self._resize_img(face))

        session.close()

        for face in faces:
            face.show()

    def show_cluster_center(self, cluster: str) -> None:
        session = self.storage.get_session()

        _, face, photo = (
            session.query(Cluster, Face, Photo)
            .filter(Cluster.name == cluster)
            .join(Face, Face.cluster_id == Cluster.id)
            .join(Photo, Face.photo_id == Photo.id)
            .first()
        )
        img = Image.open(photo.path)

        bbox = self._decode_array(face.bbox)
        face_img = img.crop(bbox)

        self._resize_img(face_img).show()

    def get_clusters_num(self) -> int:
        session = self.storage.get_session()
        c_num = session.query(Cluster).count()
        session.close()

        return c_num

    def get_clusters(self) -> None:
        session = self.storage.get_session()

        clusters = session.query(Cluster).all()
        print('Clusters:')
        for c in clusters:
            print(f'{c.name:20} | {len(c.faces)}')

        session.close()

    def rename_cluster(self, cluster: str, new_name: str) -> None:
        session = self.storage.get_session()

        cluster_orm = session.query(Cluster).filter(Cluster.name == cluster).first()
        cluster_orm.name = new_name

        session.commit()
        session.close()

    def get_photos(self, group: str) -> list[Photo]:
        return self.storage.get_photos(group)

    def present_photo(self, photo: Photo) -> None:
        faces = self.storage.get_faces(photo)

        # TODO make paths absolute ?
        photo_img = Image.open(photo.path)

        faces_img = []
        for face in faces:
            face_bbox = self._decode_array(face.bbox)
            face_img = photo_img.crop(face_bbox)
            faces_img.append(face_img)

        photo_img.show()
        for face in faces_img:
            face.show()
