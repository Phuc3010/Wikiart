from PIL import Image
from torch.utils.data import Dataset
import os

class MonetPhotoDataset(Dataset):
    def __init__(self, root_monet, root_photo, transform=None) -> None:
        super(MonetPhotoDataset, self).__init__()
        self.root_monet = root_monet
        self.root_photo = root_photo
        self.monet_imgs = os.listdir(root_monet)
        self.photo_imgs = os.listdir(root_photo)
        self.length = max(len(self.monet_imgs), len(self.photo_imgs))
        self.transforms = transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        monet_img = self.monet_imgs[idx%len(self.monet_imgs)]
        photo_img = self.photo_imgs[idx%len(self.photo_imgs)]
        monet_path = os.path.join(self.root_monet, monet_img)
        photo_path = os.path.join(self.root_photo, photo_img)
        monet = Image.open(monet_path).convert('RGB')
        photo = Image.open(photo_path).convert('RGB')
        if self.transforms:
            monet = self.transforms(monet)
            photo = self.transforms(photo)
        return photo, monet