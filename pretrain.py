import anndata as ad
import torch.hub
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tempfile import TemporaryDirectory
import os
import torch.nn.functional as F

#torch.Generator.manual_seed(41)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
DF_FILE = '/data/Collinslab/tcf7l2/embedding_anndata.h5ad'
SCRIPT_DIR = '/data/dennyal/repos/HistoGWAS_PGAN'
DINO = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')


class AnnDataset(Dataset):
    def __init__(self, pathDB, transform):
        super().__init__()
        self.adata = ad.read_h5ad(pathDB)
        self.transform = transform

    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        img_path = self.adata.obs['path'].iloc[idx]
        img = Image.open(img_path)
        img_arr = np.array(img, dtype = np.uint16).astype(np.float32)
        img_arr = img_arr / 65535.0
        transformed = self.transform(img_arr)
        transformed = transformed.repeat(3,1,1)
        one_hot = torch.from_numpy(np.asarray(self.adata.obs.iloc[idx,]["edit_id_-/-": "edit_id_WT/WT"]).astype(np.float32))
        return transformed, one_hot



def freeze_lstlayer(class_names):

    num_ftrs = 2048
    layer = nn.Linear(num_ftrs, len(class_names))
    DINO.fc = layer 

    for name, layer in DINO.named_children():
        if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']:
            for param in layer.parameters():
                param.requires_grad = False


    #DINO.to(DEVICE)




def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, out_dir, num_epochs=25):
    since = time.time()
    os.makedirs(out_dir, exist_ok=True)
    best_model_params_path = os.path.join(out_dir, 'best_model_params.pt')
    

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    print(dataset_sizes)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                _, labels = torch.max(labels.data, 1)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print("done with epoch")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model



def eval():
    freeze_lstlayer(['WT/WT', "WT/-", "-/-","CT", "TT"])
    DINO.load_state_dict(torch.load("./embeddings/model_checkpoints/best_model_params.pt"))
    DINO.eval() 
    DINO.to(DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),])
    ad_data = AnnDataset(DF_FILE, transform=transform)


    og_ad = AnnDataset(DF_FILE, transform)
    img, label = og_ad.__getitem__(0)
    with torch.no_grad():
        img.to(DEVICE)
        output = DINO(img)

    _, pred = torch.max(output, axis = 1)
    _, actual = torch.max(label, axis = 1)
    print(pred)
    print(actual)
    return output




def start(epochs):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),])
    ad_data = AnnDataset(DF_FILE, transform=transform)

    

    train_size = int(0.8 * len(ad_data))
    val_size = len(ad_data) - train_size
    print(train_size)
    print(val_size)
    print(train_size + val_size)
    print("Total images found", len(ad_data))
    dataset_sizes = {"train": train_size, "val": val_size}

    train_data, val_data = random_split(ad_data, lengths=[train_size, val_size])
    train_load = DataLoader(train_data, batch_size = 32, shuffle=True)
    val_load = DataLoader(val_data, batch_size=32, shuffle=True)
    dataloaders = {"train": train_load, "val": val_load}

    for inputs, labels in dataloaders['train']:
        print(inputs.shape)
        print(labels.shape)
        break


    print(dataloaders)

    #set up the model
    freeze_lstlayer(['WT/WT', "WT/-", "-/-","CT", "TT"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, DINO.parameters()), lr =1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    fine_tuned_dino = train_model(DINO, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes= dataset_sizes,
                     out_dir="/data/dennyal/repos/HistoGWAS_PGAN/embeddings/model_checkpoints",   num_epochs=epochs)



if __name__ == "__main__":
    eval()