import glob
import os
import shutil
import services.vision_service as vs

print("Scanning dataset to find images for every Norwood stage...")

# Collect all images from the dataset
all_images = glob.glob(
    "data/hair_density_dataset/**/*.jpg", recursive=True
) + glob.glob("data/hair_density_dataset/**/*.png", recursive=True)

found_types = {}
target_types = ["type_2", "type_4", "type_5", "type_6"]

for img_path in all_images:
  try:
    res = vs.predict_baldness_density(img_path)
    n_type = res["norwood_type"]

    if n_type in target_types and n_type not in found_types:
      out_name = f"test_samples/{n_type.upper()}_SAMPLE.jpg"
      shutil.copy(img_path, out_name)
      found_types[n_type] = (out_name, res["density_level"], res["confidence"])
      print(f"✓ Found {n_type.upper()}: {res['density_level']} ({res['confidence']}%) -> Saved to {out_name}")

    if len(found_types) == len(target_types):
      break
  except Exception:
    continue

print("\n--- Summary of Found Samples ---")
for t, (path, name, conf) in sorted(found_types.items()):
  print(f"• File: {path} | Stage: {name} | Conf: {conf}%")