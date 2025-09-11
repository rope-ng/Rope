from flask import Flask, request, jsonify
from face_swap_functions import FaceSwapProcessor

app = Flask(__name__)

face_swap_processor = None

def get_face_swap_processor():
    global face_swap_processor
    if face_swap_processor is None:
        face_swap_processor = FaceSwapProcessor()
    return face_swap_processor

@app.route('/api/swap/image', methods=['POST'])
def swap_faces():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided",
                "data": None
            }), 400
        
        if 'input' not in data:
            return jsonify({
                "status": "error", 
                "message": "Missing 'input' parameter (base64 encoded image)",
                "data": None
            }), 400
            
        if 'faces' not in data or not isinstance(data['faces'], list):
            return jsonify({
                "status": "error",
                "message": "Missing 'faces' parameter (array of base64 encoded face images)",
                "data": None
            }), 400
            
        if not data['faces']:
            return jsonify({
                "status": "error",
                "message": "At least one face image is required in 'faces' array",
                "data": None
            }), 400
        
        processor = get_face_swap_processor()
        
        result = processor.process_face_swap(
            input_image_b64=data['input'],
            source_faces_b64=data['faces'],
            parameters=data.get('parameters')
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}",
            "data": None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "success",
        "message": "Face swap API is running",
        "data": None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)