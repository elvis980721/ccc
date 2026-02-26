from PIL import Image
import torch
from torchvision import transforms, models
import os

class_names = ["大典館", "大孝館", "大忠館", "大恩館", "大成館", "大義館"]

# 載入模型
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 測試圖片
image_path = "test.jpg"  # 放一張你要測試的圖片
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    _, pred = torch.max(outputs, 1)
    print("辨識結果：", class_names[pred.item()])
