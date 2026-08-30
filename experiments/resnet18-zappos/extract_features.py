# import os.path

# import numpy as np
# import pandas as pd

import joblib
# import swifter

import torch
import torch.nn as nn
import torchvision.models as models
# import torchvision.transforms as transforms

from PIL import Image

# from common import *

from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import prepare_data

LIMIT = None
BATCH_SIZE = 32
OUTPUT_PATH = Path("output/zappos_embeddings_N50025.pkl")

def get_device():
    #nvidia GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    #Apple gpu
    if torch.backends.mps.is_available():
        return torch.device("mps")
    #cpu
    return torch.device("cpu")

class ShoeDataset(Dataset):
    #receive panda table
    def __init__(self, dataframe):
        #store table and reset index
        self.dataframe = dataframe.reset_index(drop=True)

        weights = models.ResNet18_Weights.DEFAULT
        self.transform = weights.transforms()
    #tells pytorch how many samples are in the dataset
    def __len__(self):
        return len(self.dataframe)
    #loads one image using its row number
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, row["image"], row["category"]

#create and prepare resnet18 model for feature extraction
def create_model(device):
    #load the resnet18 model with pretrained weights
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Remove the classification layer.
    # ResNet-18 will now return a 512-value feature vector instead of a 1000 object categories (previously)
    # now just return 512 visual feature eg. shape, edges, textures, colour..
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    return model

def extract_features():
    device = get_device()
    print("Using device:", device)

    dataframe = prepare_data(limit=LIMIT)
    print("Images:", len(dataframe))
    #create the dataset
    dataset = ShoeDataset(dataframe)
    #create data loader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(device)

    all_embeddings = []
    all_paths = []
    all_categories = []

    #disable gradient calculation
    with torch.inference_mode():
        # one batch at a time
        for batch_number, (images, paths, categories) in enumerate(loader, start=1):
            images = images.to(device)

            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu()

            all_embeddings.append(embeddings)
            all_paths.extend(paths)
            all_categories.extend(categories)

            print(
                f"Batch {batch_number}/{len(loader)} completed"
            )

    all_embeddings = torch.cat(all_embeddings).numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "embeddings": all_embeddings,
            "paths": all_paths,
            "categories": all_categories,
        },
        OUTPUT_PATH,
    )

    print("Embedding shape:", all_embeddings.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    extract_features()
# IMAGE_WIDTH = 224
# IMAGE_HEIGHT = 224
# CHANNELS = 3
# EMBEDDING_SIZE = 512

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def get_embedding(model, filepath):
#     try:
#         img = Image.open(filepath).convert('RGB') 
#     except:
#         print(f'image not found: {filepath}')
#         return torch.zeros(EMBEDDING_SIZE)

#     scale = transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
#     standardize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ) # default values used on imagenet
#     to_tensor = transforms.ToTensor()

#     transformed_img = standardize(to_tensor(scale(img))).unsqueeze(0)
#     transformed_img = transformed_img.to(device)

#     embedding = torch.zeros(EMBEDDING_SIZE)

#     def copy_layer(model, input, output):
#         embedding.copy_(output.data.reshape(EMBEDDING_SIZE))

#     pooling_layer = model._modules.get('avgpool')
#     attached_layer = pooling_layer.register_forward_hook(copy_layer)
#     model(transformed_img)
#     attached_layer.remove()

#     return embedding


# def get_similarity(embedding1, embedding2):
#     return nn.CosineSimilarity()(embedding1, embedding2)


# if __name__ == '__main__':
#     df = prepare_data()

#     image_name = df.iloc[0].image
#     image = load_image(image_name)

#     model = get_model(device)

#     embedding1 = get_embedding(model, get_image_path(df.iloc[1].image)).reshape((1, -1))
#     embedding2 = get_embedding(model, get_image_path(df.iloc[1000].image)).reshape((1, -1))

#     figures = {'im'+str(i): load_image(row.image) for i, row in df.iloc[[1, 1000]].iterrows()}
#     show_images(figures, 1, 2, filename='output/testimages.png')

#     cos_sim = get_similarity(embedding1, embedding2)
#     print(f'cosine similarity: {cos_sim}')


#     # todo: run in batches
#     embeddings = df['image'].swifter.apply(lambda img: get_embedding(model, get_image_path(img)))
#     embeddings = embeddings.apply(pd.Series)

#     joblib.dump(embeddings, 'output/embeddings.pkl', 9)

