import os

from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
import io
import torchvision.models as models

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3) 
# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Load the trained model
checkpoint = torch.load('best_model(1).pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for both Inception and MobileNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

app = Flask(__name__)

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        img_bytes = file.read()
        predicted_class, confidence = predict(img_bytes)
        class_names = ['Donkey', 'Horse', 'Zebra']  # Replace with actual class names
        confidence_threshold = 0.85  # Set a threshold for confidence
        
        if confidence < confidence_threshold:
            return jsonify({'class': 'None of those three classes'})

        return jsonify({'class': class_names[predicted_class]})

    return jsonify({'error': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
