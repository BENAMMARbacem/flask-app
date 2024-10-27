from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from DataPreprocessing import DataPreprocessor
from FeatureEngineering import FeatureEngineering  # Import the time module
import tensorflow as tf  # Import TensorFlow to load the .keras model

app = Flask(__name__)
CORS(app)
# Load your trained Keras model
model = tf.keras.models.load_model('./best_model.keras') 
# GET route to display "Hello, World!"
@app.route('/', methods=['GET'])
def hello():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    UPLOAD_FOLDER="input_data"
    # Define the path to save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save the file to the server
    file.save(file_path)
    print(f"File saved to: {file_path}")
     # Be cautious with large files
    data_preprocessor = DataPreprocessor(print_cost=True) 
    data_preprocessor.pipeline_noise_reduction('input_data', 'reduce_noise_output')
    data_preprocessor.pipeline_investigator_and_Patient_Audio_Separation('reduce_noise_output', 'separate_audio_output')
    
    feature_engineering = FeatureEngineering(print_cost=True) 
    feature_engineering.pipeline_dividing_audio_into_chunks('separate_audio_output', 'divided_chunks_output', 12)
    feature_engineering.pipeline_normalise_chunk_amplitude('divided_chunks_output', 'normalized_amplitude_output')
    feature_engineering.pipeline_normalise_chunk_length('normalized_amplitude_output', 'normalized_chunk_length_output', 5000)
    data = feature_engineering.pipeline_combine_mfcc_image('normalized_chunk_length_output')
    result = model.predict(data)
    
    # Determine Alzheimer's likelihood based on result threshold
    if result > 0.5:
        message = "The patient has Alzheimer’s"
    else:
        message = "The patient does not have Alzheimer’s"
    
    # Prepare a JSON response
    response = {
        "message": message,
        "filename": file.filename,
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
