from torch import nn,optim
from utils import *
from tqdm import tqdm
import torch
from torchvision.utils import save_image

def train(critic_monet, critic_photo, gen_monet, gen_photo, loader, opt_critic, opt_gen, epoch, lambda_cycle, device='cuda:0'):
    criterion = nn.MSELoss()
    loop = tqdm(loader, leave=True)

    for idx, (photo, monet) in enumerate(loop):
        monet = monet.to(device)
        photo = photo.to(device)

        fake_photo = gen_photo(monet)
        critic_photo_real = critic_photo(photo).reshape(-1)
        critic_photo_fake = critic_photo(fake_photo.detach()).reshape(-1)
        critic_photo_loss = -(torch.mean(critic_photo_real)-torch.mean(critic_photo_fake))

        fake_monet = gen_monet(photo)
        critic_monet_real = critic_monet(monet).reshape(-1)
        critic_monet_fake = critic_monet(fake_monet.detach()).reshape(-1)
        critic_monet_Loss = -(torch.mean(critic_monet_real)-torch.mean(critic_monet_fake))

        critic_loss = (critic_photo_loss + critic_monet_Loss)
        opt_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        opt_critic.step()

        critic_photo_fake_gen = critic_photo(fake_photo).reshape(-1)
        critic_monet_fake_gen = critic_monet(fake_monet).reshape(-1)
        loss_gen_photo = -torch.mean(critic_photo_fake_gen)
        loss_gen_monet = -torch.mean(critic_monet_fake_gen)

        cycle_monet = gen_monet(fake_photo)
        cycle_photo = gen_photo(fake_monet)
        cycle_monet_loss = torch.abs(monet-cycle_monet).mean()
        cycle_photo_loss = torch.abs(photo-cycle_photo).mean()
        identity_monet = gen_monet(monet)
        identity_photo = gen_photo(photo)
        identity_monet_loss = torch.abs(monet-identity_monet).mean()
        identity_photo_loss = torch.abs(photo-identity_photo).mean()
        gen_loss = (
                loss_gen_photo
                + loss_gen_monet
                + cycle_monet_loss * lambda_cycle
                + cycle_photo_loss * lambda_cycle
                + identity_monet_loss
                + identity_photo_loss
            )

        opt_gen.zero_grad()
        gen_loss.backward(retain_graph=True)
        opt_gen.step()

        if idx % 100 == 0:
            save_image(fake_photo*0.5+0.5, f"photo_{idx}.png")
            save_image(fake_monet*0.5+0.5, f"monet_{idx}.png")
            checkpoint = {
                'generator_monet': gen_monet.state_dict(),
                'generator_photo': gen_photo.state_dict(),
                'critic_monet': critic_monet.state_dict(),
                'critic_photo': critic_photo.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_critic': opt_critic.state_dict(),
                'epoch': epoch, 
                "lambda_cycle": lambda_cycle
            }
            save_checkpoint(checkpoint)
        loop.set_postfix(critic_loss=critic_loss.item(), gen_loss=gen_loss.item())