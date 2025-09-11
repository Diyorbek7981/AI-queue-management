from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base


DATABASE_URL = "postgresql://queue_user:1234@localhost:5432/queue_bd"

engine = create_engine(DATABASE_URL, echo=True)


Base = declarative_base()

session = sessionmaker(autocommit=False, autoflush=False, bind=engine)