import glob
import os
import shutil

os.makedirs('test_samples', exist_ok=True)

# 1. Grab a real Normal Hair image
normal_imgs = glob.glob('data/hair_density_dataset/**/notbald/*.*', recursive=True)
if normal_imgs:
  shutil.copy(normal_imgs[0], 'test_samples/1_NORMAL_HAIR_SAMPLE.jpg')
  print('✓ Created: test_samples/1_NORMAL_HAIR_SAMPLE.jpg (Healthy Full Hair)')

# 2. Grab a real Bald/Hair Loss image
bald_imgs = glob.glob('data/hair_density_dataset/**/bald/*.*', recursive=True)
if bald_imgs:
  shutil.copy(bald_imgs[0], 'test_samples/2_HAIR_LOSS_SAMPLE.jpg')
  print('✓ Created: test_samples/2_HAIR_LOSS_SAMPLE.jpg (Hair Loss Pattern)')

# 3. Grab Psoriasis and Alopecia images
pso_imgs = glob.glob(
    r'C:\Users\Saran Karthikeyan\OneDrive\Documents\Hair Loss ML Train\MLProjHairLoss\dataset\Disease image\psoriasis\*.*'
)
if pso_imgs:
  shutil.copy(pso_imgs[0], 'test_samples/3_PSORIASIS_SCALP_SAMPLE.jpg')
  print('✓ Created: test_samples/3_PSORIASIS_SCALP_SAMPLE.jpg (Psoriasis Scalp)')

alo_imgs = glob.glob(
    r'C:\Users\Saran Karthikeyan\OneDrive\Documents\Hair Loss ML Train\MLProjHairLoss\dataset\Disease image\alopecia-areata\*.*'
)
if alo_imgs:
  shutil.copy(alo_imgs[0], 'test_samples/4_ALOPECIA_SCALP_SAMPLE.jpg')
  print('✓ Created: test_samples/4_ALOPECIA_SCALP_SAMPLE.jpg (Alopecia Scalp)')

print(
    '\nAll test images are ready in: C:\\Users\\Saran'
    ' Karthikeyan\\hair_scalp_ai\\test_samples\\'
)