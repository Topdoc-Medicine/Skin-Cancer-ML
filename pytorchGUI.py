import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path

# model = CNNet(5)
# checkpoint = torch.load(Path("/content/weights.h5"))
# model.load_state_dict(checkpoint)

h5file =  "/content/weights.h5"

model = torch.load(h5file, map_location= torch.device('cpu'))


trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

def prediction():
  image = Image.open(Path("/content/HAM10000_images_part_1/ISIC_0024306.jpg"))

  input = trans(image)

  input = input.view(1, 3, 256,256)

  output = model(input)

  return output


# prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

finalPrediction = prediction()
print(finalPrediction[[0]])
if (finalPrediction == 0):
    print('You have been diagnosed with Melanocytic nevi. Please contact a doctor for assistance soon.')
elif (finalPrediction == 1):
    print('You have been diagnosed with Melanoma. Please contact a doctor for assistance soon.')
elif (finalPrediction == 2):
    print('You have been diagnosed with Benign keratosis-like lesions. You are healthy, but consider confirming with a specialist.')
elif (finalPrediction == 3):
    print('You have been diagnosed with Basal Cell Carcinoma. Please contact a doctor for assistance soon.')
elif (finalPrediction == 4):
    print('You have been diagnosed with Actinic Keratoses. Please contact a doctor for assistance soon.')
elif (finalPrediction == 5):
    print('You have been diagnosed with Vascular Lesions. Please contact a doctor for assistance soon.')
elif (finalPrediction == 6):
    print('You have been diagnosed with Dermatofibroma. Please contact a doctor for assistance soon.')
