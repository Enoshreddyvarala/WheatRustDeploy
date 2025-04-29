import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import torch.nn.functional as F

app = Flask(__name__)

# Ensure the static directory exists
os.makedirs('static', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
try:
    # Initialize the model
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, 6)
    
    # Load the state dict
    state_dict = torch.load("model/Final VIT.pkl", map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Define the classes
classes = ["O", "MR", "MRMS", "MS", "R", "S"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", error="No file uploaded")
        
        file = request.files["image"]
        if file.filename == '':
            return render_template("index.html", error="No file selected")
            
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = file.filename
            img_path = os.path.join("static", filename)
            file.save(img_path)
            
            try:
                # Load and preprocess the image
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    prediction = classes[predicted.item()]
                    confidence = round(confidence.item() * 100, 2)
                
                # Get suggestions based on prediction
                if prediction == "O":
                    suggestions = [
                        "Continue current management practices as they are effective",
                        "Maintain regular field monitoring (weekly checks)",
                        "Consider using this variety in future plantings"
                    ]
                elif prediction == "MR":
                    suggestions = [
                        "Monitor field every 3-4 days for disease progression",
                        "Apply preventive fungicide if weather conditions favor rust development",
                        "Maintain balanced nitrogen fertilization"
                    ]
                elif prediction == "MRMS":
                    suggestions = [
                        "Apply fungicide treatment within 48 hours",
                        "Increase field monitoring to every 2 days",
                        "Consider early harvest if infection spreads rapidly"
                    ]
                elif prediction == "MS":
                    suggestions = [
                        "Apply emergency fungicide treatment immediately",
                        "Monitor field daily for disease spread",
                        "Prepare for early harvest if conditions worsen"
                    ]
                elif prediction == "R":
                    suggestions = [
                        "Continue current management practices",
                        "Document resistance characteristics for breeding programs",
                        "Consider using this variety in breeding programs"
                    ]
                elif prediction == "S":
                    suggestions = [
                        "Apply emergency fungicide treatment immediately",
                        "Consider early harvest to minimize losses",
                        "Plan for variety change in next season"
                    ]
                
                return render_template("index.html", 
                                    prediction=prediction,
                                    confidence=confidence,
                                    image_path=img_path,
                                    suggestions=suggestions)
            
            except Exception as e:
                return render_template("index.html", error=f"Error processing image: {str(e)}")
        else:
            return render_template("index.html", error="Invalid file type")
    
    return render_template("index.html")

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)
