import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== Rotary Position Embedding (RoPE) ==========

def apply_rotary_pos_emb(x, cos, sin):
    """
    对输入张量 x（形状：[batch, seq_len, num_heads, head_dim]）应用 RoPE:
      - x[..., 0::2], x[..., 1::2] 分别处理偶数/奇数维度
      - cos, sin 形状与 x[..., 0::2] 对应
    返回: x_rotated 与 x 同形状
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rotated = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return x_rotated


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        """
        head_dim 必须为偶数
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)  # [max_seq_len, head_dim/2]
        self.register_buffer("cos", torch.cos(sinusoid_inp))
        self.register_buffer("sin", torch.sin(sinusoid_inp))
        self.max_seq_len = max_seq_len

# ========== Multi-Head Attention ==========

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1, use_rotary=True, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        self.use_rotary = use_rotary
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, model_dim]
        mask: [batch_size, 1, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]
        """
        B, q_len, _ = query.shape
        B, k_len, _ = key.shape
        # 1) 做线性映射
        q = self.q_proj(query)  # [B, q_len, model_dim]
        k = self.k_proj(key)    # [B, k_len, model_dim]
        v = self.v_proj(value)  # [B, k_len, model_dim]

        # 2) reshape => [B, num_heads, seq_len, head_dim]
        q = q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) RoPE (仅对 Q/K 应用)
        if self.use_rotary:
            # 转成 [B, seq_len, num_heads, head_dim] 再应用
            q = q.transpose(1, 2)  # => [B, q_len, num_heads, head_dim]
            k = k.transpose(1, 2)  # => [B, k_len, num_heads, head_dim]
            seq_len_q = q.size(1)
            seq_len_k = k.size(1)
            # 取前 seq_len_q/k 的 cos, sin
            cos_q = self.rotary_emb.cos[:seq_len_q].unsqueeze(0).unsqueeze(2)  # [1, seq_len_q, 1, head_dim/2]
            sin_q = self.rotary_emb.sin[:seq_len_q].unsqueeze(0).unsqueeze(2)
            cos_k = self.rotary_emb.cos[:seq_len_k].unsqueeze(0).unsqueeze(2)
            sin_k = self.rotary_emb.sin[:seq_len_k].unsqueeze(0).unsqueeze(2)

            q = apply_rotary_pos_emb(q, cos_q, sin_q)
            k = apply_rotary_pos_emb(k, cos_k, sin_k)

            # 转回 [B, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        # 4) 注意力分数 => [B, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask == 0 的地方填充 -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 5) 加权 V
        context = torch.matmul(attn, v)  # [B, num_heads, q_len, head_dim]
        # 6) reshape => [B, q_len, model_dim]
        context = context.transpose(1, 2).contiguous().view(B, q_len, self.model_dim)
        output = self.out_proj(context)
        return output

# ========== Position-wise FeedForward ==========

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ========== EncoderLayer / DecoderLayer ==========

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1, use_rotary=True, max_seq_len=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout, use_rotary, max_seq_len)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attn
        attn_out = self.self_attn(x, x, x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # FFN
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1, use_rotary=True, max_seq_len=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout, use_rotary, max_seq_len)
        self.norm1 = nn.LayerNorm(model_dim)
        self.enc_dec_attn = MultiHeadAttention(model_dim, num_heads, dropout, use_rotary=False)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, enc_dec_mask=None):
        # Self-Attn (causal)
        attn_out = self.self_attn(x, x, x, mask=self_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Cross-Attn
        attn_out = self.enc_dec_attn(x, enc_output, enc_output, mask=enc_dec_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        # FFN
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x

# ========== Encoder / Decoder 模块 ==========

class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.1, use_rotary=True, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ff_dim, dropout, use_rotary, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.1, use_rotary=True, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ff_dim, dropout, use_rotary, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, enc_output, self_mask=None, enc_dec_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, enc_dec_mask)
        x = self.norm(x)
        return x

# ========== Embedding 部分 ==========

class MidiInputEmbedding(nn.Module):
    """
    输入侧 MIDI: [pitch (离散), onset (连续), duration (连续), velocity (离散)]
    - pitch: nn.Embedding(pitch_vocab, embed_dim)
    - velocity: nn.Embedding(velocity_vocab, embed_dim)
    - onset, duration: 连续 => nn.Linear(1, embed_dim)
    最后拼接 => linear => model_dim
    """
    def __init__(self, pitch_vocab, velocity_vocab, embed_dim, model_dim):
        super().__init__()
        self.pitch_emb = nn.Embedding(pitch_vocab, embed_dim)
        self.velocity_emb = nn.Embedding(velocity_vocab, embed_dim)
        self.onset_proj = nn.Linear(1, embed_dim)
        self.duration_proj = nn.Linear(1, embed_dim)
        # 总共 4 个 embed_dim
        self.proj = nn.Linear(embed_dim * 4, model_dim)

    def forward(self, x):
        # x: [batch, seq_len, 4], 其中:
        #   x[...,0]: pitch (0 ~ pitch_vocab-1)
        #   x[...,1]: onset 连续
        #   x[...,2]: duration 连续
        #   x[...,3]: velocity (0 ~ velocity_vocab-1)
        pitch = self.pitch_emb(x[..., 0].long())
        onset = self.onset_proj(x[..., 1].unsqueeze(-1).float())
        duration = self.duration_proj(x[..., 2].unsqueeze(-1).float())
        velocity = self.velocity_emb(x[..., 3].long())
        emb = torch.cat([pitch, onset, duration, velocity], dim=-1)
        return self.proj(emb)


class MidiDecoderEmbedding(nn.Module):
    """
    输出侧(Decoder 端) MIDI: [pitch, onset, duration, velocity] 全部离散
      pitch: nn.Embedding(pitch_vocab, embed_dim)
      onset: nn.Embedding(onset_vocab, embed_dim)
      duration: nn.Embedding(duration_vocab, embed_dim)
      velocity: nn.Embedding(velocity_vocab, embed_dim)
    拼接后 => linear => model_dim
    """
    def __init__(self, pitch_vocab, onset_vocab, duration_vocab, velocity_vocab,
                 embed_dim, model_dim):
        super().__init__()
        self.pitch_emb = nn.Embedding(pitch_vocab, embed_dim)
        self.onset_emb = nn.Embedding(onset_vocab, embed_dim)
        self.duration_emb = nn.Embedding(duration_vocab, embed_dim)
        self.velocity_emb = nn.Embedding(velocity_vocab, embed_dim)
        self.proj = nn.Linear(embed_dim * 4, model_dim)

    def forward(self, x):
        # x: [batch, seq_len, 4] (pitch_idx, onset_idx, duration_idx, velocity_idx)
        pitch = self.pitch_emb(x[..., 0].long())
        onset = self.onset_emb(x[..., 1].long())
        duration = self.duration_emb(x[..., 2].long())
        velocity = self.velocity_emb(x[..., 3].long())
        emb = torch.cat([pitch, onset, duration, velocity], dim=-1)
        return self.proj(emb)

# ========== 整体 Transformer ==========

class MidiTransformer(nn.Module):
    """
    - Encoder 输入: [pitch, onset, duration, velocity]
      (pitch, velocity 离散; onset, duration 连续)
    - Decoder 输入: [pitch, onset, duration, velocity] 全部离散
    - 输出层: 全部离散 => 4 个分类头
    """
    def __init__(self,
                 pitch_vocab_in, velocity_vocab_in,
                 pitch_vocab_out, onset_vocab_out, duration_vocab_out, velocity_vocab_out,
                 embed_dim, model_dim=512, num_heads=8, ff_dim=2048,
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, max_seq_len=2048):
        super().__init__()
        # Embedding
        self.encoder_embedding = MidiInputEmbedding(
            pitch_vocab_in, velocity_vocab_in, embed_dim, model_dim
        )
        self.decoder_embedding = MidiDecoderEmbedding(
            pitch_vocab_out, onset_vocab_out, duration_vocab_out, velocity_vocab_out,
            embed_dim, model_dim
        )
        # Encoder / Decoder
        self.encoder = Encoder(num_encoder_layers, model_dim, num_heads,
                               ff_dim, dropout, use_rotary=True, max_seq_len=max_seq_len)
        self.decoder = Decoder(num_decoder_layers, model_dim, num_heads,
                               ff_dim, dropout, use_rotary=True, max_seq_len=max_seq_len)

        # 输出层：全部分类
        self.pitch_proj = nn.Linear(model_dim, pitch_vocab_out)
        self.onset_proj = nn.Linear(model_dim, onset_vocab_out)
        self.duration_proj = nn.Linear(model_dim, duration_vocab_out)
        self.velocity_proj = nn.Linear(model_dim, velocity_vocab_out)

    def forward(self, p_midi, q_midi,
                enc_mask=None, dec_mask=None, enc_dec_mask=None):
        """
        p_midi: [batch, seq_len_p, 4] => (pitch_in, onset, duration, velocity_in)
        q_midi: [batch, seq_len_q, 4] => (pitch_out, onset_out, duration_out, velocity_out)
        enc_mask: [batch, 1, seq_len_p, seq_len_p] (可选, 用于encoder屏蔽padding部分)
        dec_mask: [batch, 1, seq_len_q, seq_len_q] (自回归mask)
        enc_dec_mask: [batch, 1, seq_len_q, seq_len_p] (若需要对encoder输出也mask)
        """
        # Encoder
        enc_emb = self.encoder_embedding(p_midi)     # [B, seq_len_p, model_dim]
        enc_out = self.encoder(enc_emb, mask=enc_mask)  # [B, seq_len_p, model_dim]

        # Decoder
        dec_emb = self.decoder_embedding(q_midi)     # [B, seq_len_q, model_dim]
        dec_out = self.decoder(dec_emb, enc_out,
                               self_mask=dec_mask,
                               enc_dec_mask=enc_dec_mask)  # [B, seq_len_q, model_dim]

        # 输出分类 (pitch/onset/duration/velocity)
        pitch_logits = self.pitch_proj(dec_out)      # [B, seq_len_q, pitch_vocab_out]
        onset_logits = self.onset_proj(dec_out)      # [B, seq_len_q, onset_vocab_out]
        duration_logits = self.duration_proj(dec_out)# [B, seq_len_q, duration_vocab_out]
        velocity_logits = self.velocity_proj(dec_out)# [B, seq_len_q, velocity_vocab_out]

        return {
            "pitch": pitch_logits,
            "onset": onset_logits,
            "duration": duration_logits,
            "velocity": velocity_logits
        }

# ========== 测试 / 使用示例 ==========

if __name__ == '__main__':
    batch_size = 2
    seq_len_p = 60
    seq_len_q = 70

    # ===== 词表大小（示例） =====
    pitch_vocab_in = 128     # 输入侧 pitch
    velocity_vocab_in = 128  # 输入侧 velocity
    # 输出侧 pitch, onset, duration, velocity 全部离散
    pitch_vocab_out = 128
    onset_vocab_out = 800
    duration_vocab_out = 400
    velocity_vocab_out = 128

    embed_dim = 64
    model_dim = 512

    model = MidiTransformer(
        pitch_vocab_in, velocity_vocab_in,
        pitch_vocab_out, onset_vocab_out, duration_vocab_out, velocity_vocab_out,
        embed_dim, model_dim=model_dim, num_heads=8, ff_dim=2048,
        num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, max_seq_len=1024
    )

    # ===== 构造随机输入 =====
    # Encoder侧 p_midi: [B, seq_len_p, 4]
    #   x[...,0]: pitch_in (0~pitch_vocab_in-1)
    #   x[...,1]: onset (连续), 这儿随机 float
    #   x[...,2]: duration (连续), 随机 float
    #   x[...,3]: velocity_in (0~velocity_vocab_in-1)
    p_pitch = torch.randint(0, pitch_vocab_in, (batch_size, seq_len_p, 1))
    p_onset = (torch.rand(batch_size, seq_len_p, 1)*1000).float()
    p_duration = (torch.rand(batch_size, seq_len_p, 1)*500).float()
    p_velocity = torch.randint(0, velocity_vocab_in, (batch_size, seq_len_p, 1))
    p_midi = torch.cat([p_pitch, p_onset, p_duration, p_velocity], dim=-1)

    # Decoder侧 q_midi: [B, seq_len_q, 4] 全部离散
    #   x[...,0]: pitch_out
    #   x[...,1]: onset_out
    #   x[...,2]: duration_out
    #   x[...,3]: velocity_out
    q_pitch = torch.randint(0, pitch_vocab_out, (batch_size, seq_len_q, 1))
    q_onset = torch.randint(0, onset_vocab_out, (batch_size, seq_len_q, 1))
    q_duration = torch.randint(0, duration_vocab_out, (batch_size, seq_len_q, 1))
    q_velocity = torch.randint(0, velocity_vocab_out, (batch_size, seq_len_q, 1))
    q_midi = torch.cat([q_pitch, q_onset, q_duration, q_velocity], dim=-1)

    # 自回归 Mask (下三角)
    mask = torch.tril(torch.ones(seq_len_q, seq_len_q)).bool()
    # => [1, 1, seq_len_q, seq_len_q]
    mask = mask.unsqueeze(0).unsqueeze(0)

    out = model(p_midi, q_midi, dec_mask=mask)
    for k,v in out.items():
        print(k, v.shape)
    # 结果:
    # pitch => [B, seq_len_q, pitch_vocab_out]
    # onset => [B, seq_len_q, onset_vocab_out]
    # duration => [B, seq_len_q, duration_vocab_out]
    # velocity => [B, seq_len_q, velocity_vocab_out]
