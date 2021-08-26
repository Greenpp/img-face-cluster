from pathlib import Path

import sqlalchemy
from sqlalchemy import select
from sqlalchemy.orm import Session, session

from .models import Base, Cluster, Face, Group, Photo


class Storage:
    def __init__(
        self,
        db_location: str = 'sqlite+pysqlite:///storage/database.db',
        debug: bool = False,
    ) -> None:
        if debug:
            db_location = 'sqlite+pysqlite:///:memory:'
        else:
            Path('./storage').mkdir(exist_ok=True)

        self.engine = sqlalchemy.create_engine(db_location, echo=debug, future=True)

    def init_database(self) -> None:
        Base.metadata.create_all(self.engine)

    def get_group_id(self, name: str) -> int:
        session = self.get_session()

        group = session.query(Group).filter(Group.name == name).first()
        if not group:
            group = Group(name=name)
            session.add(group)
            session.commit()

        g_id = group.id
        session.close()

        return g_id

    def filter_new_images(self, hashes: list[str]) -> list[bool]:
        # TODO allow for image to be in multiple groups
        session = self.get_session()
        existing_hashes = session.query(Photo.hash).filter(Photo.hash.in_(hashes)).all()
        hash_set = set([h[0] for h in existing_hashes])

        session.close()

        existing_filter = list(map(lambda x: x not in hash_set, hashes))

        return existing_filter

    # def save_image(self, image: dict, faces: list[dict], group: str) -> None:
    #     with Session(self.engine) as session:
    #         group_orm = self.get_group(group, session)

    #         image['group_id'] = group_orm.id
    #         image_orm = Photo(**image)
    #         session.add(image_orm)
    #         session.commit()

    #         faces_orm = []
    #         for face in faces:
    #             face['photo_id'] = image_orm.id
    #             face_orm = Face(**face)
    #             faces_orm.append(face_orm)

    #         session.add_all(faces_orm)
    #         session.commit()

    def get_photos(self, group: str) -> list[Photo]:
        with Session(self.engine) as session:
            group_orm = session.query(Group).filter(Group.name == group).first()
            photos = session.query(Photo).filter(Photo.group_id == group_orm.id).all()

        return photos

    def get_faces(self, photo: Photo) -> list[Face]:
        with Session(self.engine) as session:
            faces = session.query(Face).filter(Face.photo_id == photo.id).all()

        return faces

    def get_session(self) -> Session:
        # TODO Split computations and database access to separate classes
        return Session(self.engine)
