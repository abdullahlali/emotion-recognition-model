import os
import torch
import torch.nn as nn
import torch.optim as optim
import h5py  # For saving the model
from torch.utils.data import DataLoader

base_dir = '/Users/abdullahali/Desktop/side_projects/emotion_recognition'

# Save the model in HDF5 format
def save_model(model, filepath):
    with h5py.File(filepath, 'w') as f:
        for name, param in model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())

def main():
    # Load datasets
    train_dataset = torch.load(os.path.join(base_dir, 'train_dataset.pth'))
    test_dataset = torch.load(os.path.join(base_dir, 'test_dataset.pth'))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

    # Define model
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # Evaluate the model and calculate accuracy
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')


    model_path = os.path.join(base_dir, 'emotion_recognition_model.h5')
    save_model(model, model_path)
    print("Model saved in HDF5 format!")


if __name__ == '__main__':  
    main()