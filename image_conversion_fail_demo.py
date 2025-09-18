from PIL import Image 

import numpy as np
import torch
import torchvision
import torchvision.transforms as T



def correct():

    img = Image.open("./DAY15/NIHB120_plate101/Images/r01c01f01p01-ch1sk1fk1fl1.png")
    img = img.resize((4,4), resample=Image.BILINEAR)
    print(img)
    arr = np.array(img, dtype=np.uint16).astype(np.float32)

    arr /= 65535.0 
    if len(arr.shape) == 2:
        print('here')
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)


    tensor = T.functional.to_tensor(arr)
    tensor = T.Normalize([0.5],[0.5])(tensor)
    print(tensor.shape)
    return(tensor)


def wrong(): 
        img = Image.open("./DAY15/NIHB120_plate101/Images/r01c01f01p01-ch1sk1fk1fl1.png")
        img = img.resize((4,4), resample=Image.BILINEAR).convert('L')
        arr = np.asarray(img.resize((4,4), resample=Image.BILINEAR))

        if len(arr.shape) == 2:
            print('here')
            arr = arr.reshape(arr.shape[0], arr.shape[1], 1)


        tensor = T.functional.to_tensor(arr)
        tensor = T.Normalize([0.5],[0.5])(tensor)
        print(tensor.shape)
        return(tensor)


if __name__ == "__main__":
    print("Correct:")
    print(correct())
    print("------")
    print("Wrong:")
    print(wrong())


