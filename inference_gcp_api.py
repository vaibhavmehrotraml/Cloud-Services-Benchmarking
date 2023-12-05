import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import time
from tqdm import tqdm
import argparse
from google.cloud import storage

# Initialize GCS client
client = storage.Client()

# Parser for command line arguments
parser = argparse.ArgumentParser(description='Process data folder path.')
parser.add_argument('-b', '--bucket', type=str, required=True, help='GCS bucket and folder path (format: bucket/folder)')
args = parser.parse_args()

# Extract bucket name and folder path
args = parser.parse_args()

data_folder = args.data_folder
bucket = client.get_bucket(bucket_name)

# Initialize EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b1')
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to run inference on a single image
def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0)
    except Exception as e:
        print(e)
        return 0

    with torch.no_grad():
        outputs = model(image)
    return torch.argmax(outputs, dim=1)

# Start timing
start_time = time.time()

# Load images from GCS and run inference
predictions = []
blobs = bucket.list_blobs(prefix=folder_path)  # List files in the specified folder
for blob in tqdm(blobs):
    if not blob.name.endswith('.csv'):
        image_bytes = blob.download_as_bytes()
        prediction = predict_image(image_bytes)
        predictions.append(prediction)

# End timing
end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / len(predictions)

print(f"Total inference time: {total_time} seconds")
print(f"Average time per image: {avg_time_per_image} seconds")

# Write inference time to a file
with open('inference_time.txt', 'a') as file:
    file.write(f"Total inference time: {total_time} seconds\n")
    file.write(f"Average time per image: {avg_time_per_image} seconds\n")
