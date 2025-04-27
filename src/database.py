import sqlite3
import os
from datetime import datetime
import hashlib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    def __init__(self):
        self.db_file = os.getenv('DATABASE_PATH', "lung_cancer_app.db")
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        # Create users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT,
                prediction TEXT,
                confidence REAL,
                report_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create admin table
        c.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default admin if not exists
        default_username = os.getenv('ADMIN_USERNAME')
        default_password = os.getenv('ADMIN_PASSWORD')  # This is just a fallback
        default_admin = (default_username, self._hash_password(default_password))
        c.execute('INSERT OR IGNORE INTO admins (username, password_hash) VALUES (?, ?)',
                 default_admin)
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_admin(self, username, password):
        """Verify admin credentials"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('SELECT password_hash FROM admins WHERE username = ?', (username,))
        result = c.fetchone()
        
        conn.close()
        
        if result and result[0] == self._hash_password(password):
            return True
        return False
    
    def add_user(self, name, email, phone, address):
        """Add a new user"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO users (name, email, phone, address)
                VALUES (?, ?, ?, ?)
            ''', (name, email, phone, address))
            user_id = c.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            # If email already exists, return the existing user's ID
            c.execute('SELECT id FROM users WHERE email = ?', (email,))
            user_id = c.fetchone()[0]
            return user_id
        finally:
            conn.close()
    
    def add_prediction(self, user_id, image_path, prediction, confidence, report_path):
        """Add a new prediction"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO predictions (user_id, image_path, prediction, confidence, report_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, image_path, prediction, confidence, report_path))
        
        conn.commit()
        conn.close()
    
    def get_all_predictions(self):
        """Get all predictions with user information"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                users.name, users.email, users.phone,
                predictions.prediction, predictions.confidence,
                predictions.created_at, predictions.image_path,
                predictions.report_path
            FROM predictions
            JOIN users ON predictions.user_id = users.id
            ORDER BY predictions.created_at DESC
        ''')
        
        predictions = c.fetchall()
        conn.close()
        
        return predictions
    
    def get_user_predictions(self, email):
        """Get predictions for a specific user"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                predictions.prediction, predictions.confidence,
                predictions.created_at, predictions.image_path,
                predictions.report_path
            FROM predictions
            JOIN users ON predictions.user_id = users.id
            WHERE users.email = ?
            ORDER BY predictions.created_at DESC
        ''', (email,))
        
        predictions = c.fetchall()
        conn.close()
        
        return predictions 