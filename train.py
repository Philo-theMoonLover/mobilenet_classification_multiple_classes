import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data Augmentation for trainning set (Optional)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  # Randomly rotate image by 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color change
    transforms.RandomVerticalFlip(),  # Randomly flip image vertically
    transforms.RandomAffine(degrees=(-30.0, 30.0), translate=(0.1, 0.1)),  # Randomly move image
    transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale but keep 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Create transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(patience=10):
    train_path = "./dataset/train"
    valid_path = "./dataset/valid"

    # Create datasets for training and validation
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=valid_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load MobileNetV3 large pre-trained model
    model = models.mobilenet_v3_large(weights=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)  # 3 is the number of classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

    num_epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training loop
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm: progress bar
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # Early stopping
        if epoch > 150:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                torch.save(model.state_dict(), 'mobilenetv3_Large.pth')
            else:
                patience_counter += 1
                print(f'Early stopping counter: {patience_counter}/{patience}')
                if patience_counter >= patience:
                    print("Early stopping due to no improvement.")
                    break


if __name__ == "__main__":
    # Train
    early_stopping = 30
    train_model(patience=early_stopping)
