import numpy as np
import onnxruntime as ort
from PIL import Image

# Load ONNX models
disease_session = ort.InferenceSession("models/disease_classifier.onnx")
baldness_session = ort.InferenceSession("models/baldness_classifier.onnx")

# Load Class Names
with open("models/disease_classes.txt", "r") as f:
  disease_classes = [line.strip() for line in f if line.strip()]

with open("models/baldness_classes.txt", "r") as f:
  baldness_classes = [line.strip() for line in f if line.strip()]


def preprocess_image(image_file):
  """Preprocess uploaded image for MobileNetV3 (224x224, Normalized)"""
  img = Image.open(image_file).convert("RGB").resize((224, 224))
  img_data = np.array(img).astype("float32") / 255.0

  # Standard ImageNet normalization
  mean = np.array([0.485, 0.456, 0.406], dtype="float32")
  std = np.array([0.229, 0.224, 0.225], dtype="float32")
  img_data = (img_data - mean) / std

  # Reshape to (Batch, Channels, Height, Width)
  img_data = np.transpose(img_data, (2, 0, 1))
  return np.expand_dims(img_data, axis=0)


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=-1)


def predict_scalp_disease(image_file):
  """Predicts the scalp disease condition and confidence percentage"""
  tensor = preprocess_image(image_file)
  outputs = disease_session.run(None, {"input": tensor})[0]
  probs = softmax(outputs[0])
  top_idx = int(np.argmax(probs))
  return {
      "condition": disease_classes[top_idx],
      "confidence": round(float(probs[top_idx]) * 100, 2),
  }


def predict_baldness_density(image_file):
  """Evaluates hair loss vs normal hair probability and maps to clinical Norwood stages."""
  tensor = preprocess_image(image_file)
  outputs = baldness_session.run(None, {"input": tensor})[0]

  # Squeeze to 1D array of logits [logit_bald, logit_normal]
  logits = np.squeeze(outputs)
  e_x = np.exp(logits - np.max(logits))
  probs = e_x / np.sum(e_x)

  # Index 0: Bald / Hair Loss, Index 1: Normal Hair
  p_hair_loss = float(probs[0])
  p_normal = float(probs[1])

  print(
      f"[DEBUG] Density Model Probs -> Hair Loss: {p_hair_loss*100:.2f}% |"
      f" Normal: {p_normal*100:.2f}%"
  )

  # If Normal Hair probability is higher, classify as early / healthy stages (Type 1 or 2)
  if p_normal > 0.60:
    norwood_type = "type_1"
    stage_name = "Type 1 — Normal Hairline (Healthy Density)"
    conf = round(p_normal * 100, 2)
  elif p_normal > 0.40:
    norwood_type = "type_2"
    stage_name = "Type 2 — Mild Temple Recession"
    conf = round(p_normal * 100, 2)
  else:
    # If Hair Loss probability is dominant, stage along Types 3 to 7 based on severity
    conf = round(p_hair_loss * 100, 2)
    if p_hair_loss < 0.65:
      norwood_type = "type_3"
      stage_name = "Type 3 — Early Pattern Thinning"
    elif p_hair_loss < 0.80:
      norwood_type = "type_4"
      stage_name = "Type 4 — Frontal & Crown Thinning"
    elif p_hair_loss < 0.90:
      norwood_type = "type_5"
      stage_name = "Type 5 — Advanced Thinning"
    elif p_hair_loss < 0.97:
      norwood_type = "type_6"
      stage_name = "Type 6 — Severe / Merged Hair Loss"
    else:
      norwood_type = "type_7"
      stage_name = "Type 7 — Extensive Hair Loss"

  return {
      "density_level": stage_name,
      "norwood_type": norwood_type,
      "confidence": conf,
  }