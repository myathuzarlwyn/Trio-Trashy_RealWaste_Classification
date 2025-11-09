from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
import cv2
import tensorflow as tf
from PIL import Image, ImageTk, ImageOps

app = Flask(__name__)
CORS(app)

# Load model
interpreter = tf.lite.Interpreter(model_path="real_waste_cnn_model.tflite")
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
    'Glass' : 'Hazardous',
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

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        
        # 1. Get the raw model output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # 2. Find the index of the highest prediction
        prediction_index = np.argmax(output)
        
        # 3. Get the waste label string (e.g., 'metal')
        waste_label = labels[prediction_index]
        
        # 4. Use the waste label string to get the bin label string (e.g., 'Hazardous')
        bin_label = WASTE_TO_BIN_MAP.get(waste_label)
        
        # 5. Get the confidence score
        confidence = float(output[0][prediction_index]) * 100

        # Check if mapping was successful
        if bin_label is None:
            return jsonify({
                'error': f"No bin mapping found for waste: {waste_label}"
            }), 400

        # Return the final bin label and confidence as JSON
        # Your frontend JavaScript will receive this object
        return jsonify({
            'prediction': bin_label,   # Recyclable  
            'waste_type': waste_label, # paper
            'bin_path': f"/static/Bins/{bin_label}Bin.png",
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# def predict():
#     file = request.files['image']
#     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)

#     interpreter.set_tensor(input_details[0]['index'], img)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])
#     prediction = np.argmax(output)
#     label = labels[prediction]

#     prediction_values = prediction
#     # This is the WASTE index (0-8)
#     predicted_waste_index = np.argmax(prediction_values)
#     predicted_bin_index = WASTE_TO_BIN_MAP.get(predicted_waste_index)

#     # Check if mapping was successful
#     if predicted_bin_index >= len(bin_labels):
#         label=f"Error: Map provided index {predicted_bin_index}\n" f"but labels file only has {len(bin_labels)} entries."
#     else:
#         # Get the bin label (e.g., "Recyclable") from the bin index
#         predicted_label = bin_labels[predicted_bin_index]
#         # lbl_result.config(text=f"Result of the waste image: {predicted_label} (Confidence: {confidence:.2f}%)")
        
#         # --- Load and display the correct dustbin image ---
#         show_dustbin_image(predicted_label)

#     return jsonify({'prediction': label}) #label

def show_dustbin_image(self, predicted_label):
        """Loads and displays the dustbin image from the specified folder."""
        
        # We need to find the correct filename (e.g., "RecyclableBin.png")
        # The `predicted_label` is "Recyclable".
        
        # Try to find the file with "Bin" suffix, or just the label name
        possible_filenames = [
            f"{predicted_label}Bin.png",
            f"{predicted_label}Bin.jpg",
            f"{predicted_label}Bin.jpeg",
            f"{predicted_label}.png",
            f"{predicted_label}.jpg",
            f"{predicted_label}.jpeg"
        ]
        
        image_path = None
        for filename in possible_filenames:
            path = os.path.join(self.dustbin_image_folder, filename)
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            # If no image is found
            self.lbl_output_image_preview.config(image=None, text=f"Image not found for:\n{predicted_label}")
            self.tk_output_image_preview = None
            return

        try:
            # Load and display the dustbin image
            bin_image = Image.open(image_path)
            
            preview_size = (self.output_img_preview_frame.winfo_width(), self.output_img_preview_frame.winfo_height())
            
            # Create a copy to resize
            preview_image = bin_image.copy()
            
            # Use thumbnail to resize maintaining aspect ratio
            preview_image.thumbnail(preview_size, Image.LANCZOS)
            
            # Create a new white background image matching the frame size
            centered_image = Image.new("RGB", preview_size, (255, 255, 255))
            
            # Calculate coordinates to paste the thumbnail in the center
            paste_x = (preview_size[0] - preview_image.width) // 2
            paste_y = (preview_size[1] - preview_image.height) // 2
            
            # Paste the thumbnail onto the white background
            # Check if the thumbnail has an alpha channel (transparency)
            if preview_image.mode == 'RGBA':
                # Use the alpha channel as the mask
                centered_image.paste(preview_image, (paste_x, paste_y), preview_image.split()[3])
            elif preview_image.mode == 'LA': # Greyscale with Alpha
                centered_image.paste(preview_image, (paste_x, paste_y), preview_image.split()[1])
            else:
                # No alpha channel (e.g., JPEG or simple RGB PNG)
                centered_image.paste(preview_image, (paste_x, paste_y)) # No mask needed
            
            self.tk_output_image_preview = ImageTk.PhotoImage(centered_image)
            
            self.lbl_output_image_preview.config(image=self.tk_output_image_preview, text="")
            
        except Exception as e:
            self.lbl_output_image_preview.config(image=None, text=f"Error loading bin image:\n{e}")
            self.tk_output_image_preview = None

if __name__ == '__main__':
    # Development mode
    if app.config["ENV"] == "development":
        app.run(host="0.0.0.0", debug=True)
    # Production mode
    else:
        app.run(host="0.0.0.0", debug=False)
