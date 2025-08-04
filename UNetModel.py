
class FiLM(nn.Module):
    
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, num_channels*2),
            nn.SiLU(),
            nn.Linear(num_channels*2, num_channels*2)
        )
    def forward(self, x, cond_vec):
  
        h = self.fc(cond_vec)  
        gamma, beta = torch.chunk(h, 2, dim=1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta  = beta.view(x.size(0), x.size(1), 1, 1)
        return gamma, beta

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None, use_film=True):
        super().__init__()
        self.use_film = use_film and (cond_dim is not None)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        if self.use_film:
            self.film1 = FiLM(cond_dim, out_ch)
            self.film2 = FiLM(cond_dim, out_ch)

    def forward(self, x, cond_vec=None):
        x = self.conv1(x)
        x = self.gn1(x)
        if self.use_film and cond_vec is not None:
            g,b = self.film1(x, cond_vec); x = g * x + b
        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)
        if self.use_film and cond_vec is not None:
            g,b = self.film2(x, cond_vec); x = g * x + b
        x = self.act(x)
        return x

class UNetCond(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, cond_method="film", num_colors=8, cond_dim=32):
        super().__init__()
        self.cond_method = cond_method
        self.num_colors = num_colors
        self.color_emb = nn.Embedding(num_colors, cond_dim)

    
        extra_in = 0
        if cond_method == "concat_rgb":
            extra_in = 3  
        elif cond_method == "concat_idx":
            extra_in = cond_dim  

        
        self.down1 = ConvBlock(in_ch + extra_in, base_ch, cond_dim=cond_dim, use_film=(cond_method=="film"))
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base_ch, base_ch*2, cond_dim=cond_dim, use_film=(cond_method=="film"))
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(base_ch*2, base_ch*4, cond_dim=cond_dim, use_film=(cond_method=="film"))
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch*4, base_ch*8, cond_dim=cond_dim, use_film=(cond_method=="film"))

    
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.up_block3 = ConvBlock(base_ch*8, base_ch*4, cond_dim=cond_dim, use_film=(cond_method=="film"))

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.up_block2 = ConvBlock(base_ch*4, base_ch*2, cond_dim=cond_dim, use_film=(cond_method=="film"))

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.up_block1 = ConvBlock(base_ch*2, base_ch, cond_dim=cond_dim, use_film=(cond_method=="film"))

        self.final = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, img, color_idx, color_rgb):
        
        B, C, H, W = img.shape
        cond_vec = self.color_emb(color_idx)  

        x = img
        if self.cond_method == "concat_rgb":
            rgb_map = color_rgb.view(B,3,1,1).expand(B,3,H,W)
            x = torch.cat([x, rgb_map], dim=1)  
        elif self.cond_method == "concat_idx":
            emb = cond_vec.view(B,-1,1,1).expand(B,cond_vec.size(1),H,W)
            x = torch.cat([x, emb], dim=1)

        
        d1 = self.down1(x, cond_vec if self.cond_method=="film" else None)
        p1 = self.pool1(d1)
        d2 = self.down2(p1, cond_vec if self.cond_method=="film" else None)
        p2 = self.pool2(d2)
        d3 = self.down3(p2, cond_vec if self.cond_method=="film" else None)
        p3 = self.pool3(d3)

        b = self.bottleneck(p3, cond_vec if self.cond_method=="film" else None)

        
        u3 = self.up3(b)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = the3 = self.up_block3(u3, cond_vec if self.cond_method=="film" else None)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_block2(u2, cond_vec if self.cond_method=="film" else None)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1, cond_vec if self.cond_method=="film" else None)

        out = self.final(u1)  
        out = torch.sigmoid(out)  
        return out
