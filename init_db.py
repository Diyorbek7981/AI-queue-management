from database import engine, Base
from model import Person

Base.metadata.create_all(bind=engine)
