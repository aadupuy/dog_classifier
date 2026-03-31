# PyTorch Dataset class
from torch.utils.data import Dataset
import os
from PIL import Image

class DogDataset(Dataset):
    def __init__(self, df, img_dir, breed_to_idx, transform=None):
        self.df = df.reset_index(drop=True) # reset index to avoid weird index issues after splitting
        self.img_dir = img_dir
        self.transform = transform
        self.breed_to_idx = breed_to_idx # cleaner than creating this mapping inside __getitem__ every time

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        breed = row['breed']
        
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB") # model expects 3 channels
        
        label = self.breed_to_idx[breed] # convert breed name to integer label using the mapping we created
        
        if self.transform:
            image = self.transform(image)
        
        return image, label