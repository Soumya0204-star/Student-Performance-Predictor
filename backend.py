from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/api/dataset')
def get_dataset():
    # Load your actual dataset
    df = pd.read_csv('student_fuzzy_dataset.csv')
    return jsonify(df.head(10).to_dict('records'))

@app.route('/api/analyze/<student_id>')
def analyze_student(student_id):
    # Your fuzzy logic analysis here
    return jsonify({
        'attendance': {'score': 85, 'category': 'Good', 'fuzzy_value': 0.78},
        'assignments': {'score': 78, 'category': 'Good', 'fuzzy_value': 0.72},
        'final_grade': 'B+',
        'confidence': 0.76
    })