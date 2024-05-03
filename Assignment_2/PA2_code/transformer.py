import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = n_embd
        self.num_heads = num_heads
        self.d_k = n_embd // num_heads  # 32

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (16 x 2 x 32 x 32) @ (16 x 2 x 32 x 32) = (16 x 2 x 32 x 32)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)     # (16 x 2 x 32 x 32)
        context = torch.matmul(attn_probs, v)   # (16 x 2 x 32 x 32) @ (16 x 2 x 32 x 32) = (16 x 2 x 32 x 32)
        return context, attn_probs

    def split_heads(self, x):
        batch_size, seq_length, n_embd = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.W_q(q))   # (16 x 2 x 32 x 32)
        k = self.split_heads(self.W_k(k))   # (16 x 2 x 32 x 32)
        v = self.split_heads(self.W_v(v))   # (16 x 2 x 32 x 32)

        context, attn_prob = self.scaled_dot_product_attention(q, k, v, mask)   # (16 x 2 x 32 x 32), (16 x 2 x 32 x 32)
        output = self.W_o(self.concat_heads(context))   # (16 x 2 x 32 x 64)
        return output, attn_prob


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_output, attn_prob = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attn_prob


class Encoder(nn.Module):
    def __init__(self, n_embd, num_heads, drop_prob, num_layers, vocab_size, block_size):
        super(Encoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model=n_embd, num_heads=num_heads, d_ff=(4 * n_embd))
                                             for _ in range(num_layers)])

    def forward(self, x):
        token_embeddings = self.token_embedding_table(x)
        positional_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=device))
        x = token_embeddings + positional_embeddings
        attention_weights_list = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, mask=None)
            attention_weights_list.append(attention_weights)
        return x, attention_weights_list
