from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms, models
import io

app = FastAPI()

# 類別名稱 (順序要和訓練時一致)
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
#執行命令: uvicorn app:app --reload --port 8001
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 讀取圖片
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # 模型預測
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    result = class_names[pred.item()]

    return {"館別": result}
