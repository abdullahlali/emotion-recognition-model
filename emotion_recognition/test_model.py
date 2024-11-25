import os
import torch
import h5py
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import torch
import h5py
import torchvision.models as models

# Load the model from HDF5
def load_model(filepath, num_classes):
    model = models.resnet18()  # Initialize the architecture
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Adjust for your dataset's number of classes

    model_dict = {}
    with h5py.File(filepath, 'r') as f:
        for name in f.keys():
            model_dict[name] = torch.tensor(f[name][()])  # Use [()] to safely read the data

    model.load_state_dict(model_dict)
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path):
    """Preprocess the image to match the model input."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5215, 0.5215, 0.5215], [0.0518, 0.0518, 0.0518])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model, class_labels):
    """Predict the class of an image."""
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return class_labels[predicted_class.item()]

def main():
    base_dir = '/Users/abdullahali/Desktop/side_projects/emotion_recognition'
    model_path = os.path.join(base_dir, 'emotion_recognition_model.h5')
    num_classes = 7  # Adjust based on your dataset
    model = load_model(model_path, num_classes)

    class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  # Update based on your dataset
    image_path = '/Users/abdullahali/Desktop/side_projects/emotion_recognition/angry_picture.jpg'  # Provide the path to an image
    predicted_label = predict(image_path, model, class_labels)
    print(f'Predicted class: {predicted_label}')


if __name__ == '__main__':
    main()