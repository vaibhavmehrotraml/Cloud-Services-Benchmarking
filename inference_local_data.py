import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process data folder path.')
parser.add_argument('-d', '--data_folder', type=str, required=True, help='Path to the data folder')

args = parser.parse_args()

data_folder = args.data_folder

image_folder_path = data_folder


model = EfficientNet.from_pretrained('efficientnet-b1')
model.eval()

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # Add batch dimension
    image = transform(image).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image)
    return torch.argmax(outputs, dim=1)

start_time = time.time()

predictions = []
for image_name in tqdm(os.listdir(image_folder_path)):
    if image_name.endswith('.csv'):
        continue
    image_path = os.path.join(image_folder_path, image_name)
    prediction = predict_image(image_path)
    predictions.append(prediction)

end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / len(predictions)

print(f"Total inference time: {total_time} seconds")
print(f"Average time per image: {avg_time_per_image} seconds")