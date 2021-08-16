from sqlalchemy import Column, Float, Integer, LargeBinary, String
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql.schema import ForeignKey

Base = declarative_base()


class Photo(Base):
    __tablename__ = 'photos'

    id = Column(Integer, primary_key=True)
    path = Column(String)
    hash = Column(String)

    faces = relationship('Face', back_populates='faces')


class Face(Base):
    __tablename__ = 'faces'

    id = Column(Integer, primary_key=True)
    probability = Column(Float)
    bbox = Column(LargeBinary)  # TODO add numpy tobytes and frombuffer as encoding
    encoding = Column(LargeBinary)

    photo_id = Column(Integer, ForeignKey('photos.id'))
    cluster_id = Column(Integer, ForeignKey('clusters.id'))

    photo = relationship('Photo', back_populates='photos')
    cluster = relationship('Cluster', back_populates='clusters')


class Cluster(Base):
    __tablename__ = 'clusters'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    faces = relationship('Face', back_populates='faces')
