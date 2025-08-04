
from PIL import Image
import torch

test_img_path = "/content/nanogon.jpeg" 
color_name = "magenta"  
ckpt_path = "/content/cond_unet_film.pt"  
ckpt = torch.load(ckpt_path, map_location=device)
color_to_idx = ckpt["color_to_idx"]
idx_to_color = ckpt["idx_to_color"]
color_to_rgb = ckpt["color_to_rgb"]
hparams = ckpt["hparams"]


model = UNetCond(
    in_ch=3, out_ch=3, base_ch=hparams["base_ch"],
    cond_method=hparams["cond_method"], num_colors=len(color_to_idx), cond_dim=32
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()


def to_rgb(im):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255,255,255,255))
    comp = Image.alpha_composite(bg, im)
    return comp.convert("RGB")

test_img = to_rgb(Image.open(test_img_path))
test_img = test_img.resize((hparams["out_size"], hparams["out_size"]), Image.BILINEAR)
x = transforms.ToTensor()(test_img).unsqueeze(0).to(device)

assert color_name in color_to_idx, f"Unknown color '{color_name}'. Available: {list(color_to_idx.keys())}"
cidx = torch.tensor([color_to_idx[color_name]], dtype=torch.long, device=device)
crgb = torch.tensor([[c / 255.0 for c in color_to_rgb[color_name]]], dtype=torch.float32, device=device)

with torch.no_grad():
    y = model(x, cidx, crgb)
    y_img = (y.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0))
    out_img = Image.fromarray((y_img * 255).astype('uint8'))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(test_img)
plt.title("Input Outline")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(out_img)
plt.title(f"Filled: {color_name}")
plt.axis("off")
plt.show()
