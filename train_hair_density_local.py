import os
import shutil
import zipfile
import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

print("[1/5] Extracting archive (7).zip from Downloads...")
zip_path = os.path.expanduser("~/Downloads/archive (7).zip")
extract_dir = "data/hair_density_dataset"

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as z:
  z.extractall(extract_dir)

# Handle nested folder structure if present
search_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
  if any(name.lower() in ["bald", "hair_loss"] for name in dirs):
    search_dir = root
    break

print(f"[2/5] Detected dataset at: {search_dir}")
print("Subfolders found:", os.listdir(search_dir))

# 2. Data Transforms & Augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(search_dir, transform=train_transforms)
classes = full_dataset.classes
print(f"Loaded {len(full_dataset)} images across classes: {classes}")

# Split into 85% Train and 15% Validation
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Apply val transforms to validation split
val_data.dataset.transform = val_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[3/5] Training on device: {device}")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 3. Model: Pre-trained MobileNetV3 (Fast & High Accuracy on CPU/GPU)
model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

# Freeze initial layers for fast transfer learning
for param in model.features[:6].parameters():
  param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# 4. Training Loop (4 Epochs)
print("[4/5] Training for 4 epochs...")
for epoch in range(4):
  model.train()
  running_loss, train_correct, train_total = 0.0, 0, 0
  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, preds = outputs.max(1)
    train_correct += preds.eq(labels).sum().item()
    train_total += labels.size(0)

  # Validation check
  model.eval()
  val_correct, val_total = 0, 0
  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, preds = outputs.max(1)
      val_correct += preds.eq(labels).sum().item()
      val_total += labels.size(0)

  val_acc = 100.0 * val_correct / val_total
  print(
      f"Epoch [{epoch+1}/4] - Train Loss: {running_loss/len(train_loader):.4f}"
      f" - Validation Accuracy: {val_acc:.2f}%"
  )

# 5. Export to ONNX
print("[5/5] Exporting model to models/baldness_classifier.onnx...")
model.eval().to("cpu")
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "models/baldness_classifier.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# Save embedded weights
m = onnx.load(onnx_path)
onnx.save_model(m, onnx_path, save_as_external_data=False)

# Save human-readable class names
clean_classes = []
for c in classes:
  if c.lower() in ["notbald", "normal", "healthy"]:
    clean_classes.append("Normal Hairline (Healthy Density)")
  elif c.lower() in ["bald", "hair_loss", "hairloss"]:
    clean_classes.append("Hair Loss / Significant Thinning")
  else:
    clean_classes.append(c.replace("_", " ").title())

with open("models/baldness_classes.txt", "w") as f:
  f.write("\n".join(clean_classes))

print("\n✓ Successfully trained and exported models/baldness_classifier.onnx!")
print("Classes configured:", clean_classes)