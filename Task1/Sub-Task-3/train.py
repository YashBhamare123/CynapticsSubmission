import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import config as c
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Encoder, DiscriminatorZ, DiscriminatorImg

data_dir = "./UTKFace/labeled"
transforms = v2.Compose([
    v2.Resize((128,128)),
    v2.ToTensor(),
    v2.Normalize(*c.stats)
])
dataset = ImageFolder(data_dir, transform= transforms)
dl = DataLoader(dataset, c.batch_size, num_workers= 0, pin_memory= True, shuffle = True)


def fit(enc, gen, discIMG, discZ, dl, opt_enc, opt_gen, opt_IMG, opt_Z, tb_step, writer):
    pass



def main():
    writer = SummaryWriter("./TB_logs")
    gen = Generator(c.num_age_cat, c.encode_size)
    enc = Encoder(c.in_channels_enc, c.encode_size)
    discIMG = DiscriminatorImg(c.in_channels_disc,)

if __name__ == "__main__":
    main()