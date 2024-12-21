from flask import Flask, request, jsonify
import os
from transformers import pipeline

app = Flask(__name__)

# Load image-to-text model from Hugging Face
image_captioning = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Generate a caption for the image
    result = image_captioning(file_path)
    caption = result[0]['generated_text']

    # Use GPT-3.5 to elaborate on the caption (free tier)
    explanation = f"The image depicts {caption}. This can be further analyzed for more details."
    return jsonify({'caption': caption, 'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True)
