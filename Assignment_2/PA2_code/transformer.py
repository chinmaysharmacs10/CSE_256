import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.d_k = n_embd // num_heads  # 32

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (16 x 2 x 32 x 32) @ (16 x 2 x 32 x 32) = (16 x 2 x 32 x 32)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-2)     # (16 x 2 x 32 x 32)
        context = torch.matmul(attn_probs, value)   # (16 x 2 x 32 x 32) @ (16 x 2 x 32 x 32) = (16 x 2 x 32 x 32)
        return context, attn_probs

    def split_heads(self, x):
        batch_size, seq_length, n_embd = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_embd)

    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.W_q(query))   # (16 x 2 x 32 x 32)
        key = self.split_heads(self.W_k(key))   # (16 x 2 x 32 x 32)
        value = self.split_heads(self.W_v(value))   # (16 x 2 x 32 x 32)

        context, attn_prob = self.scaled_dot_product_attention(query, key, value, mask)  # (16 x 2 x 32 x 32), (16 x 2 x 32 x 32)
        output = self.W_o(self.concat_heads(context))   # (16 x 2 x 32 x 64)
        return output, attn_prob
'''


class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super(AttentionHead, self).__init__()
        self.d_k = head_size
        self.W_q = nn.Linear(n_embd, head_size, bias=False)
        self.W_k = nn.Linear(n_embd, head_size, bias=False)
        self.W_v = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, mask=None):
        batch_size, seq_length, n_embd = x.shape
        query, key, value = self.W_q(x), self.W_k(x), self.W_v(x)
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(self.tril[:seq_length, :seq_length] == 0, float('-inf'))

        attention_weights = torch.softmax(attention_weights, dim=-1)
        out = torch.matmul(attention_weights, value)
        return out, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super(MultiHeadAttention, self).__init__()
        head_size = n_embd // num_heads
        self.attention_heads = nn.ModuleList([AttentionHead(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.W_o = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        outputs = []
        attention_weights = []
        for attention_head in self.attention_heads:
            out, attention_weight = attention_head(x, mask)
            outputs.append(out)
            attention_weights.append(attention_weight)
        out = torch.cat(outputs, dim=-1)
        # attention_weights = torch.stack(attention_weights, dim=0).transpose(0, 1)
        return out, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, d_ff, block_size):
        super(Block, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_embd, num_heads, block_size)
        self.feed_forward = FeedForward(n_embd, d_ff)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask):
        attn_output, attn_prob = self.multi_head_attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attn_prob


class Encoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size):
        super(Encoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.encoder_layers = nn.ModuleList([Block(n_embd=n_embd, num_heads=num_heads, d_ff=(4 * n_embd),
                                                   block_size=block_size) for _ in range(num_layers)])

    def forward(self, x):
        token_embeddings = self.token_embedding_table(x)
        positional_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=device))
        x = token_embeddings + positional_embeddings
        attention_weights_list = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, mask=None)
            attention_weights_list.extend(attention_weights)
        return x, attention_weights_list
