import torch.nn as nn
import torch
from utils import batch_add

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, in_channels, 5,2,2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels*2, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels*2, in_channels*4, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels*4, in_channels*8, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8*8*8*in_channels, out_channels)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.layers(x)
        out = self.tanh(out)
        return out




class Generator(nn.Module):
    def __init__(self, num_age_cat, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels + num_age_cat, 8*8*1024),
            nn.LeakyReLU(0.2),
        )
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 3,1, 1)
        )


    def reshape(self, x):
        return x.view(-1, 1024, 8, 8)

    def forward(self, x):
        out = self.fc(x)
        out = self.reshape(out)
        out = self.layers(out)
        return out




class DiscriminatorImg(nn.Module):
    def __init__(self, in_channels, num_age_cat, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_age_cat = num_age_cat
        self.in_channels = in_channels
        self.initial = nn.Sequential(
            nn.Conv2d(3, in_channels, 5, 2, 2),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2)
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels + self.num_age_cat, in_channels*2, 5,2,2),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels*2, in_channels * 4, 5, 2, 2),
            nn.BatchNorm2d(in_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels*4, in_channels * 8, 5, 2, 2),
            nn.BatchNorm2d(in_channels * 8),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*8*in_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def age_concat(self, x, age_one_hot, batch_size):
        plate = torch.tensor([])
        for i in range(age_one_hot.size()[1]):
            plate = torch.full((batch_size, 1, 64, 64), age_one_hot[0, 0][i])
            x = torch.cat((x, plate), dim=1)
        return x


    def forward(self, x, age_one_hot):
        out = self.initial(x)
        out = self.age_concat(out, age_one_hot, self.batch_size)
        out = self.layers(out)
        out = self.fc(out)
        out = out.view(-1,1)
        return out



class DiscriminatorZ(nn.Module):
    def __init__(self,z_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16,1)
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(-1,1)
        return out

if __name__ == "__main__":
    # Initializing the objects
    in_channels = 64
    encode_size = 50
    in_channels_disc = 16
    batch_size = 16
    age_one_hot = torch.tensor([[[0,0,0,0,1,0,0]]])
    num_age_cat = age_one_hot.size()[2]
    enc = Encoder(in_channels, encode_size)
    gen = Generator(num_age_cat, encode_size)
    discIMG = DiscriminatorImg(in_channels_disc, num_age_cat, batch_size)
    discZ = DiscriminatorZ(encode_size)

    # Simulating the training loop
    input_img = torch.randn(batch_size, 3, 128, 128)
    out_enc = enc(input_img)
    assert out_enc.size() == (batch_size,encode_size)
    age_one_hot = batch_add(age_one_hot, batch_size)
    latent_noise = torch.cat((out_enc, age_one_hot), dim = 1)
    out_gen = gen(latent_noise)
    assert out_gen.size() == (batch_size, 3, 128, 128)
    real = discIMG(input_img, age_one_hot)
    fake = discIMG(out_gen, age_one_hot)
    assert real.size() == (batch_size, 1) and fake.size() == (batch_size, 1)
    out_disc_z = discZ(out_enc)
    assert out_disc_z.size() == (batch_size,1)
    print("Success")
