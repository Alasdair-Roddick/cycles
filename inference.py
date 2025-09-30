import torch
from PIL import Image
from torchvision import transforms
from cyclegan import Generator

def load_image(path, size=256):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)

def save_image(tensor, path):
    inv = transforms.Normalize((-1,-1,-1),(2,2,2))
    img = inv(tensor.squeeze(0)).clamp(0,1)
    Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).astype("uint8")).save(path)

def stylize(input_path, ckpt_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator().to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()
    x = load_image(input_path).to(device)
    with torch.no_grad():
        y = G(x)
    save_image(y, output_path)

# usage:
# stylize("film.jpg", "checkpoints/G_F2M_50.pt", "film_monet.jpg")
