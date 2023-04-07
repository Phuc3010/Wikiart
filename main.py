from model import *
from train import train
import torch
from torch import nn, optim
import argparse
import itertools
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argumentd('--beta2', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--monet_dir', type=str, default='../input/gan-getting-started/monet_jpg')
    parser.add_argument('--photo_dir', type=str, default='../input/gan-getting-started/photo_jpg')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='./model.pt')
    
    args = parser.parse_args()
    device = args.device
    gen_monet = Generator().to(device)
    gen_photo = Generator().to(device)
    critic_monet = Critic().to(device)
    critic_photo = Critic().to(device)
    opt_gen = optim.Adam(itertools.chain(gen_monet.parameters(),gen_photo.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_critic = optim.Adam(
            itertools.chain(critic_monet.parameters(), critic_photo.parameters()), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    checkpoint = {
                'generator_monet': gen_monet.state_dict(),
                'generator_photo': gen_photo.state_dict(),
                'critic_monet': critic_monet.state_dict(),
                'critic_photo': critic_photo.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_critic': opt_critic.state_dict(),
                'epoch': 0
                }
    epochs = args.epochs
    lambda_cycle = args.lambda_cycle
    dataset, loader = get_loader(args.monet_dir, args.photo_dir, args.batch_size, shuffle=True)
    for epoch in range(epochs):
        train(critic_monet, critic_photo, gen_monet, gen_photo, loader, opt_critic, opt_gen, epoch, lambda_cycle)

