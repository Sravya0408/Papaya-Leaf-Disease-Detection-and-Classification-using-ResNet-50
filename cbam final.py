import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================
# DEVICE
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================
# CBAM MODULE
# =========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# =========================================
# RESNET-18 + CBAM
# =========================================
class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.cbam1(self.model.layer1(x))
        x = self.cbam2(self.model.layer2(x))
        x = self.cbam3(self.model.layer3(x))
        x = self.cbam4(self.model.layer4(x))

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return self.model.fc(x)

# =========================================
# DATA
# =========================================
train_dir = r"D:\papay_dataset\Papaya Leaf Disease Image Dataset\augmented"
val_dir   = r"D:\papay_dataset\Papaya Leaf Disease Image Dataset\new"

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# =========================================
# MODEL SETUP
# =========================================
model = ResNet18_CBAM(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# =========================================
# TRAINING
# =========================================
def train_model(epochs=30):

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        scheduler.step()

        # 🔹 Accuracy NOT printed
        print(f"Epoch [{epoch+1}/30] | Loss: {epoch_loss:.4f}")

    return train_losses, train_accuracies

# =========================================
# RUN TRAINING
# =========================================
train_losses, train_accuracies = train_model(epochs=30)

# =========================================
# LOSS PLOT
# =========================================
plt.figure()
plt.plot(train_losses)
plt.title("papaya_resnet18_cbam_adamw1 - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# =========================================
# ACCURACY PLOT
# =========================================
plt.figure()
plt.plot(train_accuracies)
plt.title("papaya_resnet18_cbam_adamw1 - Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()

# =========================================
# VALIDATION METRICS
# =========================================
model.eval()

correct_per_class = [0] * num_classes
total_per_class = [0] * num_classes
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(len(labels)):
            label = labels[i]
            pred = preds[i]
            total_per_class[label.item()] += 1
            if label == pred:
                correct_per_class[label.item()] += 1

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n📊 PER-CLASS VALIDATION ACCURACY")
print("=" * 50)

for i in range(num_classes):
    accuracy = 100 * correct_per_class[i] / total_per_class[i]
    print(f"{class_names[i]}: {accuracy:.2f}%")

# =========================================
# PER-CLASS PRECISION, RECALL, F1
# =========================================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

print("\n📊 PER-CLASS PRECISION | RECALL | F1-SCORE")
print("=" * 60)

for class_name in class_names:
    precision = report[class_name]["precision"] * 100
    recall = report[class_name]["recall"] * 100
    f1 = report[class_name]["f1-score"] * 100

    print(f"{class_names}")
    print(f"   Precision : {precision:.2f}%")
    print(f"   Recall    : {recall:.2f}%")
    print(f"   F1-Score  : {f1:.2f}%")
    print("-" * 40)

# =========================================
# BAR CHART
# =========================================
precisions = []
recalls = []
f1_scores = []
accuracies = []

for i, class_name in enumerate(class_names):
    accuracies.append(100 * correct_per_class[i] / total_per_class[i])
    precisions.append(report[class_name]["precision"] * 100)
    recalls.append(report[class_name]["recall"] * 100)
    f1_scores.append(report[class_name]["f1-score"] * 100)

x = range(len(class_names))

plt.figure(figsize=(10,6))
plt.bar(x, accuracies, width=0.2, label="Accuracy")
plt.bar([p + 0.2 for p in x], precisions, width=0.2, label="Precision")
plt.bar([p + 0.4 for p in x], recalls, width=0.2, label="Recall")
plt.bar([p + 0.6 for p in x], f1_scores, width=0.2, label="F1-Score")

plt.xticks([p + 0.3 for p in x], class_names, rotation=45)
plt.ylabel("Percentage (%)")
plt.title("papaya_resnet18_cbam_adamw1")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# CONFUSION MATRIX
# =========================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================================
# SAVE MODEL
# =========================================
save_path = r"D:\papay_dataset\saved_models\papaya_resnet18_cbam_adamw1.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(model.state_dict(), save_path)
print("Model saved at:", save_path)
