from sqlalchemy import Column, Float, Integer, LargeBinary, String
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql.schema import ForeignKey

CASCADE = 'CASCADE'

Base = declarative_base()


class Photo(Base):
    __tablename__ = 'photos'

    id = Column(Integer, primary_key=True)
    path = Column(String)
    hash = Column(String, unique=True)

    group_id = Column(Integer, ForeignKey('groups.id'))

    faces = relationship('Face')

    def __repr__(self) -> str:
        return f'Photo(id={self.id}, path={self.path}, hash={self.hash}, faces_num={len(self.faces)})'


class Group(Base):
    __tablename__ = 'groups'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    photos = relationship('Photo')

    def __repr__(self) -> str:
        return f'Group(id={self.id}, name={self.name}, photos_num={len(self.photos)})'


class Face(Base):
    __tablename__ = 'faces'

    id = Column(Integer, primary_key=True)
    probability = Column(Float)
    bbox = Column(LargeBinary)  # TODO add numpy tobytes and frombuffer as encoding
    encoding = Column(LargeBinary)

    photo_id = Column(Integer, ForeignKey('photos.id', ondelete=CASCADE))
    cluster_id = Column(Integer, ForeignKey('clusters.id'))

    def __repr__(self) -> str:
        return f'Face(id={self.id}, probability={self.probability}, photo={self.photo_id}, cluster={self.cluster_id})'


class Cluster(Base):
    __tablename__ = 'clusters'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    center = Column(Integer, ForeignKey('faces.id'))

    faces = relationship('Face', foreign_keys='Face.cluster_id')

    def __repr__(self) -> str:
        return f'Cluster(id={self.id}, name={self.name}, faces_num={len(self.faces)})'
