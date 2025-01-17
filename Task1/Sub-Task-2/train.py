import torch
from math import log2
from torchvision.utils import make_grid
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
from Model import Generator, Discriminator
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Hyperparameters
lr = 1e-3
batch_size = 16
img_size = 64
num_epochs = 1
latent_size = 512
in_channels = 512
lambda_gp = 10
num_steps = 5
fixed_noise = (8,latent_size,1,1)
stats = ((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))

# Helper Functions

def plot_to_tensorboard(writer, loss_gen, loss_disc, real_imgs, fake_imgs, tb_step):
    writer.add_scalar("Discriminator Loss", loss_disc, tb_step)
    with torch.no_grad():
        real_grid = make_grid(real_imgs[:8], normalize=True)
        fake_grid = make_grid(fake_imgs[:8], normalize=True)
        writer.add_image("Real", real_grid, global_step = tb_step)
        writer.add_image("Fake", fake_grid, global_step = tb_step)

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('mps')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def get_loader(img_size, device):
    transforms = v2.Compose(
        [
            v2.Resize((img_size,img_size)),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(*stats),
        ]
    )
    dataset = ImageFolder("./celebA/img_align_celeba/", transforms)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True, drop_last=True)
    length = len(dataset)
    loader = DeviceDataLoader(loader, device)
    return loader, length
# Initializations
device = get_default_device()
gen = Generator(latent_size, in_channels)
disc = Discriminator(in_channels)
load_model = True
step = 0
if load_model:
    state_dict_gen = torch.load("./Checkpoints/Gen_3.pth")
    state_dict_disc = torch.load("./Checkpoints/Disc_3.pth")
    gen.load_state_dict(state_dict_gen)
    disc.load_state_dict(state_dict_disc)
    step = 3
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.99))
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.0, 0.99))
writer = SummaryWriter("./TB_logs")
tb_step = 0
b_id=0

# Shifting the initialized models to GPU
gen = gen.to(device = device)
disc = disc.to(device = device)

# Fit Function
def fit(gen, disc, opt_gen, opt_disc, epochs, step, tb_step, device):

    while step< num_steps:
        loader, len_dataset = get_loader(4*2**step, device)
        alpha = 0
        for epoch in range(epochs):
            gen.train()
            disc.train()
            for b_id, (real,_) in enumerate(tqdm(loader)):

                # Training the Discriminator

                noise = torch.randn(batch_size, latent_size, 1, 1).to(device)
                fake = gen(noise, alpha, step)
                disc_fake = disc(fake.detach(), alpha, step)
                disc_real = disc(real, alpha, step)
                gp = gradient_penalty(disc, real, fake, alpha, step, device)
                loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp*gp +0.001*torch.mean(disc_real**2)

                opt_disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()

                # Training the Generator
                gen_fake = disc(fake, alpha, step)
                loss_gen = -(torch.mean(gen_fake))
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                if alpha<1:
                    alpha = alpha + 2*batch_size/float(len_dataset*epochs)
                    if alpha>1:
                        alpha = 1

                if (b_id%2000) == 0:
                    with torch.no_grad():
                        gen.eval()
                        gen_noise = torch.randn(fixed_noise).to(device)
                        rand_fakes = gen(gen_noise, alpha, step)
                        rand_fakes = rand_fakes*0.5 + 0.5 # To denormalize the image which is being normalized in the data loader

                        plot_to_tensorboard(writer,loss_gen.item(), loss_disc.item(),real.detach(), rand_fakes.detach(), tb_step)
                        tb_step  = tb_step+ 1
                        if step >= 4:
                            torch.save(gen.state_dict(), f"./Checkpoints/Gen_{step}v{b_id/2000}.pth")
        torch.save(gen.state_dict(), f"./Checkpoints/Gen_{step}.pth")
        torch.save(disc.state_dict(), f"./Checkpoints/Disc_{step}.pth")
        step += 1

if __name__ == "__main__":
    fit(gen, disc, opt_gen, opt_disc, num_epochs, step, tb_step, device)


