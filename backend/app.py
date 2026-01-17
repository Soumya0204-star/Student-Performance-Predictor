from flask import Flask, request, jsonify

app = Flask(__name__)

# Add CORS headers manually (no flask-cors needed)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    print("📥 Received prediction request")
    
    try:
        # Get data
        data = request.get_json()
        print(f"Data: {data}")
        
        if not data:
            return jsonify({'error': 'No data'}), 400
        
        # Extract values
        attendance = float(data.get('attendance', 0))
        internal = float(data.get('internal', 0))
        assignment = float(data.get('assignment', 0))
        
        print(f"Values: A={attendance}, I={internal}, AS={assignment}")
        
        # Simple calculation
        score = (attendance * 0.3) + (internal * 0.4) + (assignment * 0.3)
        score = round(score, 2)
        
        print(f"Score: {score}")
        
        # Determine category
        if score >= 80:
            category = "EXCELLENT"
        elif score >= 60:
            category = "GOOD"
        elif score >= 40:
            category = "AVERAGE"
        else:
            category = "NEEDS IMPROVEMENT"
        
        print(f"Category: {category}")
        
        # Return response
        return jsonify({
            'status': 'success',
            'prediction': {
                'score': score,
                'category': category
            },
            'inputs': {
                'attendance': attendance,
                'internal': internal,
                'assignment': assignment
            },
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("✅ Server running on http://127.0.0.1:5000")
    print("✅ No database, no external dependencies")
    app.run(debug=True, port=5000)