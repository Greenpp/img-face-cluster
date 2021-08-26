import hashlib
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm

from ..processor import Detector, Encoder
from ..provider import Provider
from .models import Cluster, Face, Group, Photo
from .storage import Storage

ORIENTATION_TAG = 274
EXIF_ORIENTATION = {
    1: 0,
    2: 0,
    3: 180,
    4: 180,
    5: 90,
    6: 90,
    7: -90,
    8: -90,
}


def calculate_hash(provider_out: tuple) -> tuple:
    img, path = provider_out
    h_func = hashlib.sha256()

    h_func.update(img.tobytes())
    hashed_img = h_func.hexdigest()

    return hashed_img, path


class Manager:
    def __init__(self, debug: bool = False, gpu: bool = True) -> None:
        self.storage = Storage(debug=debug)
        self.storage.init_database()

        self.detector = Detector(gpu=gpu)
        self.encoder = Encoder(gpu=gpu)
        print('Manager ready')

    def _create_provider(self, path: str, ext: list[str]) -> Provider:
        provider = Provider(path, ext)

        return provider

    def _calculate_hash(self, img: Image.Image) -> str:
        h_func = hashlib.sha256()

        h_func.update(img.tobytes())
        hashed_img = h_func.hexdigest()

        return hashed_img

    def _calculate_hash_wrapper(self, provider_out: tuple) -> tuple:
        img, path = provider_out
        hash = self._calculate_hash(img)

        return hash, path

    def _encode_array(self, arr: np.ndarray) -> bytes:
        return arr.tobytes()

    def _decode_array(self, b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32)

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        mirrored = False
        rotation = 0

        try:
            metadata = img.getexif()
            orientation_code = metadata[ORIENTATION_TAG]

            rotation = EXIF_ORIENTATION[orientation_code]
            mirrored = not bool(orientation_code % 2)
        except KeyError:
            pass

        if rotation != 0:
            img = img.rotate(-rotation, expand=True)
        if mirrored:
            img = ImageOps.mirror(img)

        img = self._resize_img(img, 720)

        return img

    def load_images(self, paths: list[str]) -> list[Image.Image]:
        return [Image.open(p) for p in paths]

    def filter_list(self, l: list, mask: list[bool]) -> list:
        if len(l) != len(mask):
            raise IndexError('Elements number does not match')

        return [l[i] if mask[i] else None for i in range(len(l))]

    def sort_data_for_processing(
        self, data: list[tuple[str, str, Image.Image]]
    ) -> list[tuple[str, str, Image.Image]]:
        return sorted(data, key=lambda x: (x[2].size[0], x[2].size[1]))

    def scan(
        self,
        path: str,
        group: str,
        extensions: list[str] = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG'],
        max_batch_size: int = 8,
    ):
        provider = self._create_provider(path, extensions)
        with Pool() as p:
            processed_img = list(
                tqdm(
                    p.imap_unordered(calculate_hash, provider),
                    total=len(provider),
                )
            )
        hashes, paths = zip(*processed_img)

        new_img_mask = self.storage.filter_new_images(hashes)
        new_img_count = sum(new_img_mask)
        print(f'Found {new_img_count} new images')

        paths = self.filter_list(paths, new_img_mask)
        hashes = self.filter_list(hashes, new_img_mask)
        images = self.load_images(paths)

        sorted_data = self.sort_data_for_processing(list(zip(paths, hashes, images)))

        img_buffer = []
        hash_buffer = []
        path_buffer = []
        for img_path, img_hash, img in tqdm(
            sorted_data,
            desc='Extracting',
            total=new_img_count,
        ):
            img = self._preprocess_image(img)

            if not img_buffer or (
                img.size[0] == img_buffer[-1].size[0]
                and img.size[1] == img_buffer[-1].size[1]
                and len(img_buffer) < max_batch_size
            ):
                img_buffer.append(img)
                hash_buffer.append(img_hash)
                path_buffer.append(img_path)
            else:
                self.process_batch(img_buffer, path_buffer, hash_buffer, group)
                img_buffer.clear()
                hash_buffer.clear()
                path_buffer.clear()
                img_buffer.append(img)
                hash_buffer.append(img_hash)
                path_buffer.append(img_path)

        self.process_batch(img_buffer, path_buffer, hash_buffer, group)

    def process_batch(
        self,
        img_batch: list[Image.Image],
        path_batch: list[str],
        hash_batch: list[str],
        group: str,
    ) -> None:
        if not img_batch:
            return

        faces, probs, boxes = self.detector.extract_faces(img_batch)

        # Filter out instances without face
        faces_idx = []
        faces_clean = []
        for f in faces:
            if f is not None:
                faces_clean.append(f)
                if faces_idx:
                    faces_idx.append(faces_idx[-1] + len(f))
                else:
                    faces_idx.append(len(f))
            else:
                if faces_idx:
                    faces_idx.append(faces_idx[-1])
                else:
                    faces_idx.append(0)
        faces_idx.pop()

        if not faces_clean:
            return  # Fix

        stacked_faces = torch.cat(faces_clean)
        stacked_embeddings = self.encoder.encode_faces(stacked_faces)
        embeddings = np.split(stacked_embeddings, faces_idx)

        group_id = self.storage.get_group_id(group)
        for path, hash, emb, prob, bbox in zip(
            path_batch, hash_batch, embeddings, probs, boxes
        ):
            img = Photo(path=str(path), hash=hash, group_id=group_id)
            if bbox is not None:
                faces_obj = [
                    Face(probability=p, bbox=b, encoding=e)
                    for p, b, e in zip(prob, bbox, emb)
                ]
            else:
                faces_obj = []

            self.save_img_data(img, faces_obj)

    def save_img_data(self, img: Photo, faces: list[Face]) -> None:
        session = self.storage.get_session()

        session.add(img)
        session.commit()
        for face in faces:
            face.photo_id = img.id

        session.add_all(faces)
        session.commit()
        session.close()

    def cluster(self, group: str, prob_threshold: float = 0.99):
        # TODO when more than one face from single photo is in the same cluster, leave most probable one ?
        session = self.storage.get_session()
        group_orm = session.query(Group).filter(Group.name == group).first()
        group_photos = session.query(Photo).filter(Photo.group_id == group_orm.id).all()
        group_photo_ids = [p.id for p in group_photos]

        group_faces = (
            session.query(Face)
            .filter(
                Face.photo_id.in_(group_photo_ids), Face.probability > prob_threshold
            )
            .all()
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

    def scan_and_cluster(self, path: str) -> None:
        # TODO ability to rescan group
        session = self.storage.get_session()

        groups_num = session.query(Group).count()
        new_name = f'g_{groups_num + 1}'

        self.scan(path, new_name)
        self.cluster(new_name)

        session.close()

    def interactive_cluster_naming(self) -> None:
        # TODO add flag to cluster if has been named before
        session = self.storage.get_session()

        clusters = session.query(Cluster).all()
        for c in clusters:
            self.show_cluster_center(c.name)

            new_name = input('Cluster name:')
            if new_name:
                self.rename_cluster(c.name, new_name)

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
            img = self._preprocess_image(img)
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
        img = self._preprocess_image(img)

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

    def get_groups(self) -> None:
        session = self.storage.get_session()

        groups = session.query(Group).all()
        print('Groups:')
        for g in groups:
            print(f'{g.name:20} | {len(g.photos)}')

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
