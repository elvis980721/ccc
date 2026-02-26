import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# === 基本設定 ===
data_dir = r"C:\dataset"
batch_size = 16
num_epochs = 10
save_path = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 圖片轉換（Resize + Tensor 化 + 資料增強）===
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# === 載入資料 ===
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ["train", "val"]
}
class_names = image_datasets["train"].classes
print("館別分類：", class_names)

# === 建立模型（使用預訓練的 ResNet18）===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 開始訓練 ===
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss, running_corrects = 0, 0

    for inputs, labels in tqdm(dataloaders["train"]):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets["train"])
    epoch_acc = running_corrects.double() / len(image_datasets["train"])
    print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

torch.save(model.state_dict(), save_path)
print("✅ 訓練完成，模型已儲存至", save_path)
