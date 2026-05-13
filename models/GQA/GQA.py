import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=32768):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / \
            (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, device=inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return cos, sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, max_seq_len=32768):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. 变形: (batch, seq_len, num_heads*head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)

        # 3. QK-Norm
        q = self.q_norm(q.permute(0, 2, 1, 3)).permute(
            0, 2, 1, 3)  # 先变成(batch,seq,head,dim)再归一化
        k = self.k_norm(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # 4. RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 5. GQA 核心：将K、V头复制以匹配Q的头数
        # 维度扩展: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
        k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
        v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # 6. 缩放点积注意力
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码 (causal mask)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # 7. 加权求和
        # (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 8. 合并多头并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)
