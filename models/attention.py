import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, t, _ = x.shape

        q = self.Wq(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(2, 3)) / (self.head_dim ** 0.5)
        att.masked_fill_(self.mask[:t, :t].bool(), -torch.inf)
        att = self.dropout(torch.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.proj(out)
