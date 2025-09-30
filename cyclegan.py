import os, glob, random
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

# ------------------------------
# Dataset
# ------------------------------
class ImageFolder(Dataset):
    def __init__(self, root, size=256):
        self.paths = glob.glob(os.path.join(root, "*"))
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx): 
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)

# ------------------------------
# Models
# ------------------------------
def conv_block(in_c, out_c, k, s, norm=True):
    layers = [nn.Conv2d(in_c,out_c,k,s,(k-1)//2)]
    if norm: layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c,c,3,1,1), nn.InstanceNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c,c,3,1,1), nn.InstanceNorm2d(c)
        )
    def forward(self,x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        model = [
            nn.Conv2d(3,64,7,1,3), nn.InstanceNorm2d(64), nn.ReLU(True),
            conv_block(64,128,3,2),
            conv_block(128,256,3,2),
        ]
        for _ in range(6): model.append(ResBlock(256))
        model += [
            nn.ConvTranspose2d(256,128,3,2,1,1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,3,7,1,3), nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self,x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        model = [
            conv_block(3,64,4,2, norm=False),
            conv_block(64,128,4,2),
            conv_block(128,256,4,2),
            conv_block(256,512,4,1),
            nn.Conv2d(512,1,4,1,1)
        ]
        self.model = nn.Sequential(*model)
    def forward(self,x): return self.model(x)

# ------------------------------
# Losses
# ------------------------------
def gan_loss(pred, target):
    return F.mse_loss(pred, target)

# ------------------------------
# Training
# ------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 256
    bs = 1

    film = ImageFolder("./data/film", size)
    monet = ImageFolder("./data/monet", size)
    loader_film = DataLoader(film, batch_size=bs, shuffle=True, drop_last=True)
    loader_monet = DataLoader(monet, batch_size=bs, shuffle=True, drop_last=True)
    it_monet = iter(loader_monet)

    G_F2M, G_M2F = Generator().to(device), Generator().to(device)
    D_F, D_M = Discriminator().to(device), Discriminator().to(device)
    opt_G = torch.optim.Adam(list(G_F2M.parameters())+list(G_M2F.parameters()), lr=2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(list(D_F.parameters())+list(D_M.parameters()), lr=2e-4, betas=(0.5,0.999))

    epochs = 50
    lambda_cyc, lambda_id = 10.0, 5.0

    for epoch in range(epochs):
        pbar = tqdm(loader_film, desc=f"Epoch {epoch+1}/{epochs}")
        for real_film in pbar:
            try: real_monet = next(it_monet)
            except StopIteration:
                it_monet = iter(loader_monet)
                real_monet = next(it_monet)
            real_film, real_monet = real_film.to(device), real_monet.to(device)

            # ----------------- Train Generators -----------------
            opt_G.zero_grad()
            fake_monet = G_F2M(real_film)
            fake_film = G_M2F(real_monet)

            recov_film = G_M2F(fake_monet)
            recov_monet = G_F2M(fake_film)

            id_film = G_M2F(real_film)
            id_monet = G_F2M(real_monet)

            valid = torch.ones_like(D_M(fake_monet))
            g_loss = (
                gan_loss(D_M(fake_monet), valid) +
                gan_loss(D_F(fake_film), valid) +
                lambda_cyc*(F.l1_loss(recov_film, real_film) + F.l1_loss(recov_monet, real_monet)) +
                lambda_id*(F.l1_loss(id_film, real_film) + F.l1_loss(id_monet, real_monet))
            )
            g_loss.backward()
            opt_G.step()

            # ----------------- Train Discriminators -----------------
            opt_D.zero_grad()
            valid = torch.ones_like(D_M(real_monet))
            fake = torch.zeros_like(valid)
            d_m_loss = (gan_loss(D_M(real_monet), valid) + gan_loss(D_M(fake_monet.detach()), fake)) * 0.5

            valid = torch.ones_like(D_F(real_film))
            fake = torch.zeros_like(valid)
            d_f_loss = (gan_loss(D_F(real_film), valid) + gan_loss(D_F(fake_film.detach()), fake)) * 0.5

            d_loss = d_m_loss + d_f_loss
            d_loss.backward()
            opt_D.step()

            pbar.set_postfix(g=float(g_loss.item()), d=float(d_loss.item()))

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(G_F2M.state_dict(), f"checkpoints/G_F2M_{epoch+1}.pt")

if __name__ == "__main__":
    main()
