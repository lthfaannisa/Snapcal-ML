import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
model_path = 'cal_food.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming the model expects input shape [1, 640, 640, 3]
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])

# Label for 30 food classes
labels = [
    "ayam-bakar : 216 kkal","ayam-goreng : 246 kkal","bakso : 202 kkal","bandeng : 233 kkal","bawal : 113 kkal",
    "bubur-ayam : 155 kkal","gado-gado : 132 kkal","gorengan : 228 kkal","gulai-kambing : 125 kkal",
    "lele-goreng : 105 kkal","mie-ayam : 175 kkal","nasi-goreng : 130 kkal","nasi-putih : 180 kkal",
    "opor-ayam : 163 kkal","rawon : 119 kkal","rendang : 195 kkal","roti : 264 kkal",
    "sambal-goreng-kentang : 102 kkal","sate : 161 kkal","seblak : 131 kkal","sosis : 172 kkal",
    "soto-ayam : 130 kkal","tahu : 78 kkal","telor-balado : 202 kkal","telor-goreng : 153 kkal",
    "tempe : 193 kkal","tempe-orek : 210 kkal","terong-balado : 97 kkal","tumis-kangkung : 20 kkal",
    "tumis-tauge : 86 kkal"
    ]

def preprocess_image(image: Image.Image):
    image = image.resize(input_size)
    image = np.array(image).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def predict(image: Image.Image):
    preprocessed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]
    confidence = prediction[predicted_label_index]
    return predicted_label, confidence

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(image_file).convert('RGB')
    predicted_label, confidence = predict(image)
    
    return jsonify({
        "predicted_label": predicted_label,
        "confidence": float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
