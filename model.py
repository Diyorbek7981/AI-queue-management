from database import Base
from sqlalchemy import Column, Integer, Float, DateTime


class Person(Base):
    __tablename__ = "queue_people"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    enter_time = Column(DateTime, nullable=True)
    wait_time = Column(Float, default=0)
    exit_time = Column(DateTime, nullable=True)
    service_time = Column(Float, default=0)

    def __str__(self):
        return (
            f"Track_id={self.track_id}"
        )
