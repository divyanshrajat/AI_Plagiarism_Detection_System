"""SQLite-backed user store with password hashing.

This module provides user authentication and management using SQLite.
"""
from pathlib import Path
from typing import Optional, Dict, List
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
import sqlite3
import os
from datetime import datetime


_SECRET = os.getenv("USER_STORE_SECRET") or os.getenv("FLASK_SECRET") or "dev-secret-key-for-tokens"

_TOKEN_MAX_AGE = 60 * 60 * 24

DB_PATH = Path("data/users.db")

def get_db():
    """Get SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize users table."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username: str, password: str, role: str = "student", email: str = None) -> bool:
    """Create a new user."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, email) VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), role, email)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
        
    if check_password_hash(user['password_hash'], password):
        return {"username": user["username"], "role": user["role"]}
    return None

def get_user(username: str) -> Optional[Dict]:
    """Get user by username."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
    return {"username": user["username"], "role": user["role"], "email": user["email"]}

def list_users() -> List[Dict]:
    """List all users."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username, role, created_at FROM users ORDER BY created_at")
    users = cursor.fetchall()
    conn.close()
    
    result = []
    for u in users:
        result.append({
            "username": u["username"], 
            "role": u["role"], 
            "created_at": u["created_at"]
        })
    return result

def set_role(username: str, role: str) -> bool:
    """Set user role."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success

def delete_user(username: str) -> bool:
    """Delete a user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success

def _get_serializer():
    return URLSafeTimedSerializer(_SECRET)

def generate_reset_token(username: str) -> str:
    """Generate password reset token."""
    s = _get_serializer()
    return s.dumps({"username": username})

def verify_reset_token(token: str, max_age: int = _TOKEN_MAX_AGE) -> Optional[str]:
    """Verify password reset token."""
    s = _get_serializer()
    try:
        data = s.loads(token, max_age=max_age)
        return data.get("username")
    except Exception:
        return None

def reset_password(username: str, new_password: str) -> bool:
    """Reset user password."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (generate_password_hash(new_password), username)
    )
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success

