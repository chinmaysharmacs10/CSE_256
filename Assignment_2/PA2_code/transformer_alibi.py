import math

import torch
from torch import nn


def get_relative_positions(seq_len):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, drop_prob, causal):
        super().__init__()
        self.causal = causal
        self.num_heads = num_heads
        self.d_k = math.sqrt(n_embd)
        self.dropout = nn.Dropout(drop_prob)
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.key_query_value = nn.Linear(n_embd, 3 * n_embd, bias=False)

    def forward(self, x, block_size):
        batch_size, seq_len, n_embd = x.shape

        key, query, value = self.key_query_value(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        bias = (self.m * get_relative_positions(seq_len)).unsqueeze(0)

        score = (torch.matmul(query, key) / self.d_k) + bias

        mask = torch.tril(torch.ones(1, 1, block_size, block_size))
        if self.causal:
            score = score.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        attn = torch.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.dropout(out)

        attention_heads = torch.split(attn, split_size_or_sections=1, dim=1)
        attention_weights = [head_tensor.squeeze(1) for head_tensor in attention_heads]

        return out, attention_weights


class FeedForward(nn.Module):
    def __init__(self, n_embd, expansion_factor, drop_prob):
        super().__init__()
        d_hidden = n_embd * expansion_factor
        self.fc_1 = nn.Linear(n_embd, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.dropout(x)
        return x


class ALiBiBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, expansion_factor, causal, drop_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, expansion_factor, drop_prob)
        self.alibi_multi_head_attention = ALiBiMultiHeadAttention(n_embd, num_heads, block_size, drop_prob, causal)

    def forward(self, x, block_size):
        residual_x = x
        x, attention_weights = self.alibi_multi_head_attention(x, block_size)
        x = self.layer_norm_1(x + residual_x)
        residual_x = x
        x = self.feed_forward(x)
        x = self.layer_norm_2(x + residual_x)
        return x, attention_weights


class ALiBiEncoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size, drop_prob, expansion_factor, causal):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.encoder_layers = nn.ModuleList([ALiBiBlock(n_embd, num_heads, block_size, expansion_factor, causal,
                                                        drop_prob) for _ in range(num_layers)])

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.token_embedding_table(x)
        attention_weights_list = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, self.block_size)
            attention_weights_list.extend(attention_weights)
        return x, attention_weights_list


class ALiBiDecoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size, expansion_factor, causal):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.decoder_layers = nn.ModuleList([ALiBiBlock(n_embd, num_heads, block_size, expansion_factor, causal,
                                                        drop_prob=0.0) for _ in range(num_layers)])
        self.fc = nn.Linear(n_embd, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y, block_size):
        x = self.token_embedding_table(x)

        attention_weights_list = []
        for layer in self.decoder_layers:
            x, attention_weights = layer(x, block_size)
            attention_weights_list.extend(attention_weights)

        x = self.fc(x)

        batch_size, seq_length, vocab_size = x.shape
        logits = x.view(batch_size * seq_length, vocab_size)
        targets = y.view(batch_size * seq_length)
        loss = self.loss_fn(logits, targets)

        return loss, attention_weights_list
