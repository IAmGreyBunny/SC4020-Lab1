import os.path

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

import torchvision.models as models

import pandas as pd

IMAGE_ROOT = Path(
    "data/zappos/ut-zap50k-images/ut-zap50k-images"
)
def prepare_data(limit=None):
    image_paths = sorted(IMAGE_ROOT.rglob("*.jpg"))

    # if limit is not None:
    #     image_paths = image_paths[:limit]
        
    #make random sample rather than the first ... sample
    if limit is not None:
        image_paths = (
            pd.Series(image_paths)
            .sample(n=min(limit, len(image_paths)), random_state=42)
            .tolist()
        )

    records = []

    for path in image_paths:
        relative_path = path.relative_to(IMAGE_ROOT)
        parts = relative_path.parts

        records.append({
            "image": str(path),
            "filename": path.name,
            "category": parts[0],
            "subcategory": parts[1],
            "brand": parts[2],
        })

    return pd.DataFrame(records)

# def prepare_data():
#     df = pd.read_csv('./dataset/styles.csv', on_bad_lines='skip')
#     df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
#     df = df.reset_index(drop=True)
#     df[[os.path.isfile(i) for i in df['image']]]

#     return df

def get_image_path(filename):
    path = Path(filename)

    if path.exists():
        return str(path)

    return str(IMAGE_ROOT / filename)

# def get_image_path(filename):
#     return f'./dataset/images/{filename}'



def load_image(filename):
    return cv2.imread(get_image_path(filename))




def summarize_model(model):
    from torchsummary import summary
    summary(resnet, (CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))


def get_model(device='cpu'):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.to(device)
    return resnet


def show_images(images, rows = 1, cols=1, figsize=(12, 12), filename='show_images.png'):
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=figsize)
    
    for index, name in enumerate(images):
        axes.ravel()[index].imshow(cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB))
        axes.ravel()[index].set_title(name)
        axes.ravel()[index].set_axis_off()
        
    plt.tight_layout() 
    plt.savefig(filename)


def show_recommendations(input_img_path, recommendations, filename='recommendations.png'):
    input_img = Image.open(input_img_path)
    plt.imshow(input_img)

    figures = {'im' + str(i): Image.open(get_image_path(i)) for i in recommendations}
    
    fig, axes = plt.subplots(2, 5, figsize=(8,8) )
    for index,name in enumerate(figures):
        axes.ravel()[index].imshow(figures[name])
        axes.ravel()[index].set_title(name)
        axes.ravel()[index].set_axis_off()
    plt.tight_layout()
    plt.savefig(filename)
