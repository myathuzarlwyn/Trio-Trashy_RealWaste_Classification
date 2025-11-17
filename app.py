from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
import cv2
import tensorflow as tf
from PIL import Image, ImageOps

app = Flask(__name__)
CORS(app) 

# Load model
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_final_14Nov_2.tflite") ## # Latest model by Mimi ("mobilenetv2_final_14Nov_2.tflite")
# interpreter = tf.lite.Interpreter(model_path="real_waste_cnn_model_13Nov2025_tm.tflite") #Teachable Machine trained model ("real_waste_cnn_model_13Nov2025_tm.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
dustbin_image_folder = r"D:\AIS\Trio-Trashy_RealWaste_Classification\static\Bins"
labels = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'] 
bin_labels = ['Recyclable', 'Hazardous', 'Food', 'Residual']
IMG_SIZE = 224

# --- Waste Labels ---
# 0: 🔁 Cardboard
# 1: 🍌 Food Organics
# 2: 🧪 Glass
# 3: 🪙 Metal 
# 4: 🤖 Miscellaneous Trash
# 5: 📰 Paper  
# 6: 🥫 Plastic 
# 7: 🔋 Textile Trash
# 8: 🌱 Vegetation

# --- Dustbin Labels ---
# 0 = Recyclable
# 1 = Hazardous
# 2 = Food
# 3 = Residual

# --- Waste Category (0-8) to Bin Type (0-3) Map ---        
WASTE_TO_BIN_MAP = {
    'Cardboard' : 'Recyclable',
    'Food Organics' : 'Food',
    'Glass' : 'Recyclable',
    'Metal' : 'Hazardous',
    'Miscellaneous Trash' : 'Residual',
    'Paper' : 'Recyclable',
    'Plastic' : 'Residual',
    'Textile Trash' : 'Residual',
    'Vegetation' : 'Food'
}
    
@app.route('/')
def home():
    return render_template('index.html')

# === NEW ROUTE FOR THE TEAM PAGE ===
@app.route('/our_team')
def our_team():
    return render_template('our_team.html')

# === NEW ROUTE FOR THE PROJECT PAGE ===
@app.route('/about_project')
def about_project():
    return render_template('about_project.html')

@app.route('/predict', methods=['POST'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

        #Set Input: Give the interpreter the uploaded image 
        interpreter.set_tensor(input_details[0]['index'], img) 
        interpreter.invoke() #Run the model to get a prediction
        
        # 1. Get the raw model output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # 2. Find the index of the highest prediction
        prediction_index = np.argmax(output)
        
        # 3. Get the waste label string (e.g., 'paper')
        waste_label = labels[prediction_index]
        
        # 4. Use the waste label string to get the bin label string (e.g., 'Hazardous')
        bin_label = WASTE_TO_BIN_MAP.get(waste_label)
        
        # 5. Get the confidence score
        confidence = float(output[0][prediction_index]) * 100

        print("Array length: " + str(len(output)))
        print("prediction_index : " + str(prediction_index))
        print("waste_label : " + waste_label)
        print("bin_label : " + bin_label)
        print("confidence : " + str(confidence))

        # Check if mapping was successful
        if bin_label is None:
            return jsonify({
                'error': f"No bin mapping found for waste: {waste_label}"
            }), 400

        # Return the final bin label and confidence as JSON
        return jsonify({
            'prediction': bin_label,   # Recyclable  
            'waste_type': waste_label, # paper
            'bin_path': f"/static/Bins/{bin_label}Bin.png",
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    # Development mode
    if app.config["ENV"] == "development":
        app.run(host="0.0.0.0", debug=True)
    # Production mode
    else:
        app.run(host="0.0.0.0", debug=False)
