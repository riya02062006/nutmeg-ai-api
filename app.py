from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# TFLite model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_nutmeg_model.tflite")

# Load model once when server starts
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Nutmeg TFLite model loaded successfully!")


def predict_nutmeg(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Send image to TFLite model
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Class order: 0 = Bad, 1 = Good
    bad_probability = float(prediction[0])
    good_probability = float(prediction[1])

    good_percentage = good_probability * 100
    bad_percentage = bad_probability * 100

    # 70% threshold
    if good_probability >= 0.70:
        decision = "GOOD"
    else:
        decision = "BAD"

    return decision, good_percentage, bad_percentage


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Nutmeg AI Sorting API"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    try:
        image_file = request.files["image"]
        image = Image.open(image_file)

        decision, good_percentage, bad_percentage = predict_nutmeg(image)

        return jsonify({
            "decision": decision,
            "good_percentage": round(good_percentage, 2),
            "bad_percentage": round(bad_percentage, 2)
        })

    except Exception as e:
        print("Prediction error:", str(e))

        return jsonify({
            "error": "Unable to process image"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )
