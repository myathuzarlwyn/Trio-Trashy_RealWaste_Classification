import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
import warnings

# 1. Silence the TFLite warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*_INTERPRETER_DELETION_WARNING.*")

app = Flask(__name__)
CORS(app) 

# Load model
model_file_path = os.path.join("model_files", "mobilenetv2_fixed_FP16.tflite")
interpreter = tf.lite.Interpreter(model_path=model_file_path) ### Latest model by Mimi ("mobilenetv2_final_14Nov_2.tflite")
# interpreter = tf.lite.Interpreter(model_path="real_waste_cnn_model_13Nov2025_tm.tflite") #Teachable Machine trained model ("real_waste_cnn_model_13Nov2025_tm.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
dustbin_image_folder = r"\static\Bins"
labels = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'] 
bin_labels = ['Recyclable', 'Hazardous', 'Food', 'General']
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
# 3 = General

# --- Waste Category (0-8) to Bin Type (0-3) Map ---        
WASTE_TO_BIN_MAP = {
    'Cardboard' : 'Recyclable',
    'Food Organics' : 'Food',
    'Glass' : 'Recyclable',
    'Metal' : 'Hazardous',
    'Miscellaneous Trash' : 'General',
    'Paper' : 'Recyclable',
    'Plastic' : 'General',
    'Textile Trash' : 'General',
    'Vegetation' : 'Food'
}

CLASS_METRICS = {
    'Cardboard': {'precision': 0.93, 'recall': 0.91, 'f1': 0.92},
    'Food Organics': {'precision': 0.92, 'recall': 0.89, 'f1': 0.90},
    'Glass': {'precision': 0.85, 'recall': 0.97, 'f1': 0.90},
    'Metal': {'precision': 0.94, 'recall': 0.86, 'f1': 0.89},
    'Miscellaneous Trash': {'precision': 0.78, 'recall': 0.73, 'f1': 0.76},
    'Paper': {'precision': 0.95, 'recall': 0.92, 'f1': 0.93},
    'Plastic': {'precision': 0.86, 'recall': 0.88, 'f1': 0.87},
    'Textile Trash': {'precision': 0.82, 'recall': 0.96, 'f1': 0.88},
    'Vegetation': {'precision': 0.97, 'recall': 0.97, 'f1': 0.97}
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

        class_stats = CLASS_METRICS.get(waste_label, {})
        class_precision = float(class_stats.get('precision', 0)) * 100
        class_recall = float(class_stats.get('recall', 0)) * 100
        class_f1 = float(class_stats.get('f1', 0)) * 100

        print(f"Raw Model Output: {output}")
        print("Array length: " + str(len(output)))
        print("prediction_index : " + str(prediction_index))
        print("waste_label : " + waste_label)
        print("bin_label : " + bin_label)
        print("confidence : " + str(confidence))
        print("dustbin_image_folder: " + dustbin_image_folder)        
        print("class_precision: " + str(class_precision))
        print("class_recall: " + str(class_recall))
        print("class_f1: " + str(class_f1))

        return jsonify({
            'prediction': bin_label,
            'waste_type': waste_label,
            'bin_path': f"/static/Bins/{bin_label}Bin.png",
            'confidence': f"{confidence:.2f}%",
            # Return specific stats for the predicted class
            'class_precision': f"{class_precision:.2f}",
            'class_recall': f"{class_recall:.2f}",
            'class_f1': f"{class_f1:.2f}"
        })

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



# --- Run the app ---
# ---------------------
if __name__ == '__main__':
    app.run(host="127.0.0.1", debug=False)