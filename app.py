
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Allow your browser interface to access the API
CORS(app)

# Model location
MODEL_PATH = BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_nutmeg_model.keras")

# Load model once when the API starts
model = load_model(MODEL_PATH)

print("Nutmeg AI model loaded successfully!")


def predict_nutmeg(image):
    """
    Predict GOOD/BAD using the trained MobileNetV2 model.
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to the model input size
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # AI prediction
    prediction = model.predict(image_array, verbose=0)[0]

    # Class 0 = BAD
    # Class 1 = GOOD
    bad_probability = float(prediction[0])
    good_probability = float(prediction[1])

    # Convert to percentages
    bad_percentage = bad_probability * 100
    good_percentage = good_probability * 100

    # Final decision
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

        # Open uploaded image
        image = Image.open(image_file)

        # Run AI prediction
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

    # Railway/Render/etc. provide PORT automatically.
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
