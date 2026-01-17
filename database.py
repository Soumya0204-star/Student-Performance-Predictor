"""
Database module for storing student records and predictions
"""

import sqlite3
from datetime import datetime

class StudentDatabase:
    """SQLite database for student performance records"""
    
    def __init__(self, db_name='students.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create students table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            student_id TEXT UNIQUE,
            attendance REAL,
            internal_marks REAL,
            assignment_score REAL,
            predicted_score REAL,
            performance_category TEXT,
            prediction_date TIMESTAMP,
            notes TEXT
        )
        ''')
        
        # Create prediction history table (for tracking changes)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            attendance REAL,
            internal_marks REAL,
            assignment_score REAL,
            predicted_score REAL,
            category TEXT,
            predicted_at TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_name)
    
    def save_student(self, student_data):
        """
        Save or update a student record
        
        Args:
            student_data (dict): Student information including:
                - student_name
                - student_id
                - attendance
                - internal_marks
                - assignment_score
                - predicted_score
                - performance_category
                - notes (optional)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if student exists
        cursor.execute(
            "SELECT id FROM students WHERE student_id = ?",
            (student_data['student_id'],)
        )
        existing = cursor.fetchone()
        
        # Add timestamp
        student_data['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if existing:
            # Update existing student
            cursor.execute('''
            UPDATE students SET
                student_name = ?,
                attendance = ?,
                internal_marks = ?,
                assignment_score = ?,
                predicted_score = ?,
                performance_category = ?,
                prediction_date = ?,
                notes = ?
            WHERE student_id = ?
            ''', (
                student_data['student_name'],
                student_data['attendance'],
                student_data['internal_marks'],
                student_data['assignment_score'],
                student_data['predicted_score'],
                student_data['performance_category'],
                student_data['prediction_date'],
                student_data.get('notes', ''),
                student_data['student_id']
            ))
        else:
            # Insert new student
            cursor.execute('''
            INSERT INTO students 
            (student_name, student_id, attendance, internal_marks, 
             assignment_score, predicted_score, performance_category, 
             prediction_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                student_data['student_name'],
                student_data['student_id'],
                student_data['attendance'],
                student_data['internal_marks'],
                student_data['assignment_score'],
                student_data['predicted_score'],
                student_data['performance_category'],
                student_data['prediction_date'],
                student_data.get('notes', '')
            ))
        
        # Save to history
        cursor.execute('''
        INSERT INTO prediction_history
        (student_id, attendance, internal_marks, assignment_score,
         predicted_score, category, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_data['student_id'],
            student_data['attendance'],
            student_data['internal_marks'],
            student_data['assignment_score'],
            student_data['predicted_score'],
            student_data['performance_category'],
            student_data['prediction_date']
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def get_all_students(self):
        """Get all student records"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            student_name,
            student_id,
            attendance,
            internal_marks,
            assignment_score,
            predicted_score,
            performance_category,
            prediction_date,
            notes
        FROM students
        ORDER BY prediction_date DESC
        ''')
        
        students = []
        columns = [description[0] for description in cursor.description]
        
        for row in cursor.fetchall():
            student = dict(zip(columns, row))
            students.append(student)
        
        conn.close()
        return students
    
    def get_student(self, student_id):
        """Get a specific student by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM students WHERE student_id = ?
        ''', (student_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def delete_student(self, student_id):
        """Delete a student record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        cursor.execute('DELETE FROM prediction_history WHERE student_id = ?', (student_id,))
        
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    
    def get_statistics(self):
        """Get overall statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM students')
        total = cursor.fetchone()[0]
        
        cursor.execute('''
        SELECT 
            COUNT(*) as total,
            AVG(predicted_score) as avg_score,
            AVG(attendance) as avg_attendance,
            AVG(internal_marks) as avg_internal,
            AVG(assignment_score) as avg_assignment
        FROM students
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_students': stats[0],
            'average_score': stats[1] or 0,
            'average_attendance': stats[2] or 0,
            'average_internal': stats[3] or 0,
            'average_assignment': stats[4] or 0
        }

# Create a global database instance
db = StudentDatabase()

if __name__ == "__main__":
    # Test the database
    print("Database initialized successfully!")
    print("Tables created: students, prediction_history")