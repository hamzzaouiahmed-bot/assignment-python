from database import SessionLocal
from models import User

def get_all_users():
    session = SessionLocal()
    try:
        return session.query(User).all()
    finally:
        session.close()

def find_user_by_username(username: str):
    session = SessionLocal()
    try:
        return session.query(User).filter(User.username == username).first()
    finally:
        session.close()

def insert_user(username: str, email: str, age: int):
    session = SessionLocal()
    try:
        user = User(username=username, email=email, age=age)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def update_user(user_id: int, username: str = None, email: str = None, age: int = None):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        if username is not None:
            user.username = username
        if email is not None:
            user.email = email
        if age is not None:
            user.age = age
        session.commit()
        session.refresh(user)
        return user
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
