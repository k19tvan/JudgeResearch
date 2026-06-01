# auth.py
import datetime
import re
import bcrypt
import jwt

SECRET_KEY = "your-very-secret-key"  # In production, load this from environment variables
ALGORITHM = "HS256"
PASSWORD_MIN_LENGTH = 8

def validate_password(password: str):
    errors = []
    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append("Password must be at least 8 characters long.")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit (0-9).")
    if re.search(r"\s", password):
        errors.append("Password must not contain whitespace characters.")
    return errors

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)