from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import face_recognition
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Welcome to face matching API</h1>"

@app.route("/match", methods=["POST"])
def match():
    if request.method == "POST":
        data = request.json
        if 'original_image' in data and 'test_images' in data:
            try:
                # Download the original image from the URL
                response = requests.get(data['original_image'])
                if response.status_code == 200:
                    # Convert the image content to PIL Image object
                    original_image = Image.open(BytesIO(response.content))
                else:
                    return jsonify({"error": "Failed to download original image"}), 400

                test_images = []
                for test_image_url in data['test_images']:
                    # Download the test image from each URL
                    response = requests.get(test_image_url)
                    if response.status_code == 200:
                        # Convert the image content to PIL Image object
                        test_image = Image.open(BytesIO(response.content))
                        test_images.append(test_image)
                    else:
                        return jsonify({"error": f"Failed to download test image: {test_image_url}"}), 400

                # Convert PIL Image objects to numpy arrays
                original_face = np.array(original_image)
                original_face_encoding = face_recognition.face_encodings(original_face)[0]
                # response will contain all the image links that are matched with original image
                response = []
                for i in range(len(test_images)):
                    unknown_face = np.array(test_images[i])
                    unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
                    result = face_recognition.compare_faces([original_face_encoding], unknown_face_encoding)
                    if result[0]:
                        response.append(data["test_images"][i])
                return jsonify(response), 200
            except Exception as e:
                return jsonify({"error": f"Unable to process images: {str(e)}"}), 400
        else:
            return jsonify({"error": "original_image and test_images fields are required in the data"}), 400
    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    app.run()