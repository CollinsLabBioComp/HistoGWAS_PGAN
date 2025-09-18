from torch.utils.data import Dataset
from PIL import Image
import anndata
import torch
#import torchvision
import pdb
import numpy as np
#import torchvision.transforms as Transforms
#from ..utils.image_transform import NumpyResize, NumpyToTensor


tissue_list = {
    'Lung': [0, 1, 2, 3, 5, 6],
    'Artery_Aorta': [0, 1, 2, 3, 4, 5, 6, 10],
    'Pancreas': [0,2, 4],
    'Spleen': [0, 1, 2, 3, 5],
    'Breast_Mammary_Tissue': [0, 1, 3, 4, 8],
    'Adipose_Subcutaneous': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Colon_Transverse': [0, 1, 2, 3, 5, 6, 7, 8],
    'Artery_Tibial': [1, 2, 3, 4, 5],
    'Stomach': [0, 2, 3, 4, 5, 6, 7, 8, 10],
    'Esophagus_Mucosa': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Esophagus_Muscularis': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Muscle_Skeletal': [1, 2, 3, 4, 5, 6, 7],
    'Skin_Sun_Exposed_Lower_leg': [0, 1, 2, 3, 4, 5, 6, 7],
    'Thyroid': [0, 1, 3, 2, 4, 5, 6],
}



class AnnDataset(Dataset):
    def __init__(self, pathDB, tissue, transform):

        self.adata = anndata.read_h5ad(pathDB)
        # choice = np.random.choice(len(self.adata), 1000000, replace=False)
        # self.adata = self.adata[choice]
        # self.adata = self.adata[self.adata.obs['leiden_0.5'].isin(cluster_i)]
        self.transform = transform
        self.return_attrib = False
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.adata.X[idx]).float()
        img_path = self.adata.obs['path'].iloc[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, x


#if __name__=='__main__':

    #TESTER SCRIPT
    #pdb.set_trace()
    # pathDB = '/data/Collinslab/tcf7l2/organoid_anndata.h5ad'
    # transformlist = [NumpyResize((4,4)),
    #                     NumpyToTensor(),
    #                    Transforms.Normalize([0.5], [0.5])]
    # transform = Transforms.Compose(transformlist)

    # dataset = AnnDataset(pathDB,'Organoid',  transform)
    # loader  = torch.utils.data.DataLoader(dataset)

    # for item, data in enumerate(loader, 0):

    #     inputs_real = data[0]
    #     embs = data[1]
    #     print(inputs_real)
    #     print('----------------')



    #img, x = dataset.__getitem__(0)
    #print(img)
    #print(x)
    #pdb.set_trace()

