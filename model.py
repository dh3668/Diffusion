import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def exists(x):
    return x is not None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, dim_out, 3, padding=1)
    )


def Downsample(dim, dim_out):
    return nn.Conv1d(dim, dim_out, 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale
    

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift         # scaling & shifting using time emb
        
        x = self.act(x)
        return self.dropout(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(2)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B, self.heads, -1, L), qkv)

        q = q.softmax(dim=-2)   # [B, h, d, L]
        k = k.softmax(dim=-1)   # [B, h, d, L]
        # v.shape = [B, h, e, L]

        q = q * self.scale

        context = torch.matmul(k, v.transpose(-1,-2))  # [B, h, d, e]
        
        out = torch.matmul(context.transpose(-1,-2), q) # [B, h, d, L]
        out = out.reshape(B, -1, L)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B, self.heads, -1, L), qkv)
        # q.shape = [B, h, d, I], k.shape = [B, h, d, L], v.shape = [B, h, d, L]
        q = q * self.scale

        sim = torch.matmul(q.transpose(-1,-2), k)   # [B, h, I, L]
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(-1,-2))    # [B, h, I, d]

        out = out.reshape(B, -1, L)
        return self.to_out(out)


class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc = LinearAttention(dim, heads, dim_head)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc(h)
        return h + x
    

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc = Attention(dim, heads, dim_head)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc(h)
        return h + x
    

class DownBlock(nn.Module):
    def __init__(self, dim, dim_in, dim_out, heads=4, dim_head=32, dropout=0., last=False):
        super().__init__()
        self.time_dim = dim * 4

        self.resnetblock1 = ResnetBlock(dim_in, dim_in, time_emb_dim=self.time_dim, dropout=dropout)
        self.resnetblock2 = ResnetBlock(dim_in, dim_in, time_emb_dim=self.time_dim, dropout=dropout)
        self.linattnblock = LinearAttentionBlock(dim_in, heads, dim_head)
        self.downsample = Downsample(dim_in, dim_out) if not last else nn.Conv1d(dim_in, dim_out, 3, padding=1)

    def forward(self, x, t):
        h1 = self.resnetblock1(x, t)
        h2 = self.resnetblock2(h1, t)
        h2 = self.linattnblock(h2)
        out = self.downsample(h2)
        return out, h1, h2


class MidBlock(nn.Module):
    def __init__(self, dim, mid_dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        self.time_dim = dim * 4

        self.resnetblock1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=self.time_dim, dropout=dropout)
        self.attnblock = AttentionBlock(mid_dim, heads, dim_head)
        self.resnetblock2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=self.time_dim, dropout=dropout)

    def forward(self, x, t):
        x = self.resnetblock1(x, t)
        x = self.attnblock(x)
        return self.resnetblock2(x, t)
    

class UpBlock(nn.Module):
    def __init__(self, dim, dim_in, dim_out, heads=4, dim_head=32, dropout=0., last=False):
        super().__init__()
        self.time_dim = dim * 4

        self.resnetblock1 = ResnetBlock(dim_in + dim_out, dim_in, time_emb_dim=self.time_dim, dropout=dropout)
        self.resnetblock2 = ResnetBlock(dim_in + dim_out, dim_in, time_emb_dim=self.time_dim, dropout=dropout)
        self.linattnblock = LinearAttentionBlock(dim_in, heads, dim_head)
        self.upsample = Upsample(dim_in, dim_out) if not last else nn.Conv1d(dim_in, dim_out, 3, padding=1)

    def forward(self, x, h1, h2, t):
        x = self.resnetblock1(torch.cat((x, h2), dim=1), t)
        x = self.resnetblock2(torch.cat((x, h1), dim=1), t)
        x = self.linattnblock(x)
        return self.upsample(x)


class Unet(nn.Module):
    def __init__(self, dim=16):
        super(Unet, self).__init__()
        time_dim = dim * 4

        self.init_conv = nn.Conv1d(1, dim, 7, padding=3)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.down1 = DownBlock(dim, dim_in = dim, dim_out = dim*2)
        self.down2 = DownBlock(dim, dim_in = dim*2, dim_out = dim*4)
        self.down3 = DownBlock(dim, dim_in = dim*4, dim_out = dim*8)
        self.down4 = DownBlock(dim, dim_in = dim*8, dim_out = dim*16, last=True)
        
        self.mid = MidBlock(dim, mid_dim = dim*16)

        self.up1 = UpBlock(dim, dim_in = dim*16, dim_out = dim*8)
        self.up2 = UpBlock(dim, dim_in = dim*8, dim_out = dim*4)
        self.up3 = UpBlock(dim, dim_in = dim*4, dim_out = dim*2)
        self.up4 = UpBlock(dim, dim_in = dim*2, dim_out = dim, last=True)

        self.final_res_block = ResnetBlock(dim*2, dim)
        self.final_conv = nn.Conv1d(dim, 1, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)


        x, h1, h2 = self.down1(x, t)
        x, h3, h4 = self.down2(x, t)
        x, h5, h6 = self.down3(x, t)
        x, h7, h8 = self.down4(x, t)

        x = self.mid(x, t)

        x = self.up1(x, h7, h8, t)
        x = self.up2(x, h5, h6, t)
        x = self.up3(x, h3, h4, t)
        x = self.up4(x, h1, h2, t)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
