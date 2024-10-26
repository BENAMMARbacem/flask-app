from flask import Flask, request, jsonify
from flask_cors import CORS
import time  # Import the time module

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Print the filename
    print("Received file:", file.filename)
    
    # You can also read the file contents if needed
    file_content = file.read()  # This reads the entire file content
    print("File content:", file_content)  # Be cautious with large files

    # Simulate a delay of 3 seconds
    time.sleep(3)

    # Prepare a JSON response
    response = {
        "message": "The patient has Alzheimer’s",
        "filename": file.filename,
        "file_size": len(file_content)  # Optionally, include the size of the file
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)