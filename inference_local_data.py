import os
import time
import json
import torch
import argparse
import pandas as pd

from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Allow Image.Io to load truncated files as well.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Process data folder path.')
parser.add_argument('-d', '--data_folder', type=str, required=True, help='Path to the data folder')
parser.add_argument('-i', '--iterations', type=str, required=True, help='Number of iterations over the data')

args = parser.parse_args()

data_folder = args.data_folder
iterations = int(args.iterations)
image_folder_path = data_folder


model = EfficientNet.from_pretrained('efficientnet-b1')
model.eval()

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        # Add batch dimension
        image = transform(image).unsqueeze(0) 
    except Exception as e:
        print(e)
        return 0
    
    with torch.no_grad():
        outputs = model(image)
    return torch.argmax(outputs, dim=1)

predictions = []
prediction_times = {}

start_time = time.time()
for i in range(iterations):
    for image_name in tqdm(os.listdir(image_folder_path)):
        if image_name.endswith('.csv'):
            continue
        image_path = os.path.join(image_folder_path, image_name)
        time_image = time.time()
        prediction = predict_image(image_path)
        prediction_times[f'{image_name}_{i}'] = time.time() - time_image
        predictions.append(prediction)

end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / len(predictions)

print(f"Total inference time: {total_time} seconds")
print(f"Average time per image: {avg_time_per_image} seconds")

with open(f'inference_time_{iterations}iterations.txt', 'a') as file:
        file.write(f"Total inference time: {total_time} seconds")
        file.write(f"Average time per image: {avg_time_per_image} seconds")

with open(f'inference_times_local_{iterations}iterations.json', 'w') as fp:
    json.dump(prediction_times, fp)
