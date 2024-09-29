import os
import time
from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_model_v2(model_path, num_classes):
    print("Loading MobileNetV2...")
    # Initialize MobileNetV2 model with specific number of classes
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Switch the model to evaluation mode

    return model


def load_model_v3(model_path, num_classes):
    print("Loading MobileNetV3...")
    # Initialize MobileNetV3 model with specific number of classes
    model = models.mobilenet_v3_large()
    model.classifier[num_classes] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Switch the model to evaluation mode

    return model


def predict_with_threshold(model, image_path, preprocess, class_names, threshold=None):
    count = 0
    start_time = time.time()

    # Initialize counters for each class and for "Unknown"
    class_counts = {class_name: 0 for class_name in class_names}
    class_counts['Unknown'] = 0

    for file in os.listdir(image_path):
        if file.endswith((".jpg", ".png")):
            image = Image.open(os.path.join(image_path, file)).convert('RGB')
            count += 1

            image = preprocess(image).unsqueeze(0)

            # Predictions
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probabilities, 1)

                print("\nFile:", file)
                # print(probabilities)

                if max_prob.item() < threshold:
                    print("predicted: Unknown")
                    class_counts['Unknown'] += 1
                    continue

                predicted_class = class_names[predicted.item()]
                print("predicted:", predicted_class)
                class_counts[predicted_class] += 1

    # Calculate the average time for each prediction
    avg_time = (time.time() - start_time) / count
    print("Avg time per image:", avg_time)

    # Print classification results for all classes
    print("\nClassification Results:")
    for class_name, class_count in class_counts.items():
        print(f"{class_name}: {class_count}")


if __name__ == "__main__":
    # Inference
    class_names = ['class_01', 'class_02', 'class_03']  # classes name
    threshold = 0.8

    # Load MobileNetV2
    # model = load_model_v2('./mobilenetv2_3_classes.pth', num_classes=len(class_names))

    # Load MobileNetV3
    model = load_model_v3("mobilenetv3_Large.pth", num_classes=len(class_names))

    # Define transformation for input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize (h,w)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_path = "path/to/test/data"

    predict_with_threshold(model, image_path, preprocess, class_names, threshold)
