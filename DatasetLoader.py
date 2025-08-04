
class PolygonColorDataset(Dataset):
    def __init__(self, base_dir, records, split="train", color_to_rgb=None, augment=True, out_size=128, add_synthetic=False, synthetic_factor=2):
        self.base_dir = Path(base_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.out_size = out_size
        self.color_to_rgb = color_to_rgb or {}
        self.add_synthetic = add_synthetic and (split == "train")
        self.synthetic_factor = synthetic_factor
        self.inputs_dir = self.base_dir / split / "inputs"
        self.outputs_dir = self.base_dir / split / "outputs"
        self.records = list(records)
        if self.add_synthetic and self.synthetic_factor > 1:
            self.records = self.records * self.synthetic_factor
        colors = sorted(self.color_to_rgb.keys())
        self.color_to_idx = {c:i for i,c in enumerate(colors)}
        self.idx_to_color = {i:c for c,i in self.color_to_idx.items()}
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def _paired_affine(self, img_in, img_out):
       
        angle = random.uniform(-20, 20)
        translate = (random.uniform(-5, 5), random.uniform(-5, 5))
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-8, 8)
        img_in = TF.affine(img_in, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=transforms.InterpolationMode.BILINEAR, fill=(255,255,255))
        img_out = TF.affine(img_out, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=transforms.InterpolationMode.BILINEAR, fill=(255,255,255))
        if random.random() < 0.5:
            img_in = TF.hflip(img_in)
            img_out = TF.hflip(img_out)
        if random.random() < 0.2:
            img_in = TF.vflip(img_in)
            img_out = TF.vflip(img_out)
        return img_in, img_out

    def __getitem__(self, idx):
        rec = self.records[idx]
        cin = self.inputs_dir / rec["input_polygon"]
        cout = self.outputs_dir / rec["output_image"]
        color_name = rec["colour"]

  
        def to_rgb(im):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255,255,255,255))
            comp = Image.alpha_composite(bg, im)
            return comp.convert("RGB")

        inp = to_rgb(Image.open(cin))
        out = to_rgb(Image.open(cout))

        inp = inp.resize((self.out_size, self.out_size), Image.BILINEAR)
        out = out.resize((self.out_size, self.out_size), Image.BILINEAR)

        if self.augment:
            inp, out = self._paired_affine(inp, out)

        inp_t = self.to_tensor(inp)  
        out_t = self.to_tensor(out)  

        color_idx = self.color_to_idx[color_name]
        color_rgb = self.color_to_rgb[color_name]  
        color_rgb_t = torch.tensor([c/255.0 for c in color_rgb], dtype=torch.float32)  # [3]

        sample = {
            "inp": inp_t,
            "target": out_t,
            "color_idx": torch.tensor(color_idx, dtype=torch.long),
            "color_rgb": color_rgb_t,
            "color_name": color_name
        }
        return sample


BATCH_SIZE = 16 
OUT_SIZE = 128

train_ds = PolygonColorDataset(base_dir, train_map, split="training", color_to_rgb=color_to_rgb, augment=True, out_size=OUT_SIZE, add_synthetic=True, synthetic_factor=4)
val_ds   = PolygonColorDataset(base_dir, val_map,   split="validation", color_to_rgb=color_to_rgb, augment=False, out_size=OUT_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

len(train_ds), len(val_ds), len(train_loader), len(val_loader)