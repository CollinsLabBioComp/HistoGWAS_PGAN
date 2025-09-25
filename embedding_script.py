import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
import anndata as ad



# Load in all System Vars
IMAGE_DIR = '/data/Collinslab/tcf7l2/nyscf-organoid-images-processed/'
SCRIPT_DIR = '/data/dennyal/repos/HistoGWAS_PGAN'
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
resnet50.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(DEVICE)

#Transforms 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# We need to convert from uint16 to float
def preprocess(image_path):
    img = Image.open(image_path)
    arr = np.array(img, dtype = np.uint16).astype(np.float32)
    arr /= 65535.0
    transform_arr = transform(arr)
    img.close()
    return transform_arr


#actual embedding time
def embed(image_arr):
    with torch.no_grad():
        output = resnet50(image_arr)

    return output


#tester script for debug
def test_script(file_path):
    overall = {}
    image_dict = {}
    embeddings = []
    idx = 0
    relative_path = os.path.relpath(file_path, SCRIPT_DIR)
    input_tensor = preprocess(relative_path)
    input_tensor = input_tensor.repeat(3,1,1).unsqueeze(0)
    embedding = embed(input_tensor)
    image_dict['file'] = file_path
    overall[idx] = image_dict
    embedding = embedding.squeeze(0)
    embedding = np.asarray(embedding)
    embeddings.append(embedding)
    create_anndata(embeddings, idx, overall)


def create_anndata(embeddings, i, image_dict):
    np_embeddings = np.array(embeddings)
    print(image_dict)
    df = pd.DataFrame.from_dict(image_dict, orient='index')
    embed_ad = ad.AnnData(X=np_embeddings, obs = df)
    embed_ad.write_h5ad(f'./embeddings/embedding{i}.h5ad', compression = 'gzip')
    

def main():
    image_dict = {}
    embeddings = []
    for root, _, files in os.walk(IMAGE_DIR, topdown=True):
        relative_path = os.path.relpath(root, SCRIPT_DIR)
        for i, file in enumerate(files):
            if file.endswith('.png'):
                relative_path = os.path.join(relative_path, file)
                try:
                    img_tensor = preprocess(image_path=relative_path)
                except Exception as e:
                    print(f"Error opening image {abs_path}: {e}")
                    raise SystemExit("Issue opening images")
                
                input_tensor = img_tensor.repeat(3,1,1).unsqueeze(0)
                abs_path = os.path.join(root, file)
                gpu_tensor = input_tensor.to(DEVICE)
                embedding = embed(gpu_tensor)
                embedding_arr = np.asarray(embedding.squeeze(0))
                embeddings.append(embedding_arr)
                image_dict[i] = abs_path


                
                if len(embeddings) >= 10000:
                    create_anndata(embeddings, i, image_dict)
                    embeddings = []

            

        
if __name__ == "__main__":
    test = '/data/Collinslab/tcf7l2/nyscf-organoid-images-processed/NIHB120/DAY15/NIHB120_plate101/Images/r01c01f01p01-ch1sk1fk1fl1.png'
    test_script(test)
