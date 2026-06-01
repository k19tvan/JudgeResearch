# auth.py
import datetime
import os
import re
import bcrypt
import jwt

SECRET_KEY = "your-very-secret-key"  # In production, load this from environment variables
ALGORITHM = "HS256"
PASSWORD_MIN_LENGTH = 9
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 32
USERNAME_ALLOWED_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
USERNAME_BLOCKLIST_ENV = "USERNAME_BLOCKLIST"

def load_username_blocklist():
    raw = os.getenv(USERNAME_BLOCKLIST_ENV, "")
    return [term.strip().lower() for term in raw.split(",") if term.strip()]

def validate_username(username: str):
    errors = []
    if len(username) < USERNAME_MIN_LENGTH or len(username) > USERNAME_MAX_LENGTH:
        errors.append(
            f"Username must be {USERNAME_MIN_LENGTH}-{USERNAME_MAX_LENGTH} characters long."
        )
    if not USERNAME_ALLOWED_PATTERN.match(username):
        errors.append("Username may include letters, numbers, dots, underscores, and dashes only.")
    blocklist = load_username_blocklist()
    if blocklist and any(term in username.lower() for term in blocklist):
        errors.append("Username violates community standards.")
    return errors

def validate_password(password: str):
    errors = []
    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append("Password must be longer than 8 characters.")
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
