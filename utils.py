from torch.utils.data import DataLoader, dataset
from torchvision.transforms import transforms
from dataset import MonetPhotoDataset
import torch

def get_loader(monet_dir, photo_dir, batch_size, shuffle=True):
    transform=transforms.Compose(
    [transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = MonetPhotoDataset(monet_dir, photo_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    return dataset, loader

def save_checkpoint(state, filename='./model.pt'):
    torch.save(state, filename)

def load_checkpoint(critic_monet, critic_photo, gen_monet, gen_photo, opt_gen, opt_critic, lr,
                    path='../input/modelcycle/model (3).pt'):
    checkpoint = torch.load(path)
    gen_monet.load_state_dict(checkpoint['generator_monet'])
    gen_photo.load_state_dict(checkpoint['generator_photo'])
    critic_monet.load_state_dict(checkpoint['critic_monet'])
    critic_photo.load_state_dict(checkpoint['critic_photo'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_critic.load_state_dict(checkpoint['opt_critic'])
    for param in opt_critic.param_groups:
        param['lr'] = lr
    for param in opt_gen.param_groups:
        param['lr'] = lr
    return checkpoint
