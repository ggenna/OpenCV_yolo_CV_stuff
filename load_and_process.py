import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_image_path = './IMG-20240813-WA0008.jpeg'
test_image = Image.open(test_image_path)
#test_image.show()

transform = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
  ])

test_image_tensor = transform(test_image)
print(test_image_tensor)


# transformations for data aumentation
img_flipped=transforms.functional.hflip(test_image)
img_rotate=transforms.functional.rotate(test_image,45)
img_cropped=transforms.functional.crop(test_image,50,50,200,200)

# Visualize results
#fig, ax = plt.subplots(1, figsize=(12, 10))
#ax.imshow(img_flipped)
#plt.savefig("esempioflipped.jpg", bbox_inches="tight")

img_flipped.show()
img_rotate.show()
img_cropped.show()