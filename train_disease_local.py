import os
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

dataset_path = r"C:\Users\Saran Karthikeyan\OneDrive\Documents\Hair Loss ML Train\MLProjHairLoss\dataset\Disease image"

print(f"[1/4] Loading Scalp Disease dataset from: {dataset_path}")

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(dataset_path, transform=img_transforms)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
num_classes = len(dataset.classes)

print(f"Detected {num_classes} scalp disease classes: {dataset.classes}")

# Save class labels
with open("models/disease_classes.txt", "w") as f:
  f.write("\n".join(dataset.classes))

# 2. Model: Pretrained MobileNetV3 (Fast on CPU)
device = torch.device("cpu")
model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

# Freeze early layers
for param in model.features[:6].parameters():
  param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# 3. Train for 4 epochs (~60-90 seconds on CPU)
print("[2/4] Training Scalp Disease model on CPU...")
for epoch in range(4):
  model.train()
  running_loss, correct, total = 0.0, 0, 0
  for images, labels in loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, preds = outputs.max(1)
    correct += preds.eq(labels).sum().item()
    total += labels.size(0)

  print(
      f"Disease Epoch [{epoch+1}/4] - Loss: {running_loss/len(loader):.4f} -"
      f" Accuracy: {100.0 * correct / total:.2f}%"
  )

# 4. Export to standalone ONNX
print("[3/4] Exporting to models/disease_classifier.onnx...")
model.eval().to("cpu")
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "models/disease_classifier.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

m = onnx.load(onnx_path)
onnx.save_model(m, onnx_path, save_as_external_data=False)
print("[4/4] ✓ Successfully trained and exported models/disease_classifier.onnx!")