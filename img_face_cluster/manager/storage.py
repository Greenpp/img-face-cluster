from pathlib import Path

import sqlalchemy
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

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
            Path('./cache').mkdir(exist_ok=True)

        self.engine = sqlalchemy.create_engine(db_location, echo=debug, future=True)

    def init_database(self) -> None:
        Base.metadata.create_all(self.engine)

    def get_group(self, name: str, session: Session) -> Group:
        group = session.query(Group).filter(Group.name == name).first()
        if not group:
            group = Group(name=name)
            session.add(group)
            session.commit()

        return group

    def filter_new_images(self, hashes: list[str]) -> list[bool]:
        # TODO allow for image to be in multiple groups
        querry = select(Photo.hash).where(Photo.hash.in_(hashes))
        with Session(self.engine) as session:
            existing_hashes = session.execute(querry).all()
        hash_set = set(existing_hashes)

        existing_filter = map(lambda x: x not in hash_set, hashes)

        return list(existing_filter)

    def save_image(self, image: dict, faces: list[dict], group: str) -> None:
        with Session(self.engine) as session:
            group_orm = self.get_group(group, session)

            image['group_id'] = group_orm.id
            image_orm = Photo(**image)
            session.add(image_orm)
            session.commit()

            faces_orm = []
            for face in faces:
                face['photo_id'] = image_orm.id
                face_orm = Face(**face)
                faces_orm.append(face_orm)

            session.add_all(faces_orm)
            session.commit()
