# PathMNIST-EfficientNet
"Achieving 94.11% accuracy on PathMNIST using EfficientNet-B0 transfer learning and custom PyTorch architecture." 
# PathMNIST Classification using EfficientNet-B0 & PyTorch

A deep learning project implementing Transfer Learning with **EfficientNet-B0** to classify pathological tissue images from the **PathMNIST** dataset (MedMNIST).

## 📊 Results
* **Test Accuracy:** `94.11%`
* **Model:** EfficientNet-B0 (Pre-trained on ImageNet, fine-tuned)
* **Optimizer:** AdamW (`lr=0.0001`)
* **Loss Function:** CrossEntropyLoss

---

## 🛠️ Code Implementation


```python
### 1. Model Setup
import torch
import torchvision.models
import torch.nn as nn

model = torchvision.models.efficientnet_b0(weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=in_features, out_features=256, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=256, out_features=64, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=64, out_features=9, bias=True)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


### 2. Data Transforms & Dataloaders

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

data_flag = 'pathmnist'
info = INFO[data_flag]
data_class = getattr(medmnist, info['python_class'])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = data_class(split='train', transform=train_transform, download=True)
test_dataset = data_class(split='test', transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


### 3. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device).squeeze()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  epoch_loss = running_loss / len(train_loader)
  print(f"Epoch {epoch + 1} / {num_epochs} , loss: {epoch_loss:0.4f}")

### 4. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device).squeeze()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy:{accuracy:0.2f}%")
