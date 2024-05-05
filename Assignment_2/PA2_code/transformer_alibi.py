import math

import torch
from torch import nn


def get_relative_positions(seq_len: int) -> torch.tensor:
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
        self.scale = math.sqrt(n_embd)
        self.dropout = nn.Dropout(drop_prob)
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.kqv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        if causal:
            self.register_buffer("mask", torch.tril(torch.ones(1, 1, block_size, block_size)))

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)

        bias = (self.m * get_relative_positions(seq_len)).unsqueeze(0)
        # bias.shape == (1, num_heads, seq_len, seq_len)

        score = (torch.matmul(query, key) / self.scale) + bias
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        attn = torch.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, expansion_factor, drop_prob):
        super().__init__()
        d_hidden = n_embd * expansion_factor
        self.fc1 = nn.Linear(n_embd, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.relu(self.fc1(x))
        out = self.dropout(self.fc2(x))
        return out


class ALiBiTransformerLayer(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, expansion_factor, causal, drop_prob):
        super().__init__()
        self.ffn_norm = nn.LayerNorm(n_embd)
        self.attn_norm = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, expansion_factor, drop_prob)
        self.attn = ALiBiMultiHeadAttention(n_embd, num_heads, block_size, drop_prob, causal)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ALiBiEncoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size, drop_prob, expansion_factor, causal):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.encoder_layers = nn.ModuleList(
            [ALiBiTransformerLayer(n_embd, num_heads, block_size, expansion_factor, causal, drop_prob)
             for _ in range(num_layers)]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.token_embedding_table(x)
        # attention_weights_list = []
        for layer in self.encoder_layers:
            x = layer(x)
            # attention_weights_list.extend(attention_weights)
        return x, []


class ALiBiDecoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size, expansion_factor, causal) :
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.decoder_layers = nn.ModuleList(
            [ALiBiTransformerLayer(n_embd, num_heads, block_size, expansion_factor, causal, drop_prob=0.0)
             for _ in range(num_layers)])
        self.fc = nn.Linear(n_embd, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.token_embedding_table(x)

        # attention_weights_list = []
        for layer in self.decoder_layers:
            x = layer(x)
            # attention_weights_list.extend(attention_weights)

        x = self.fc(x)

        batch_size, seq_length, vocab_size = x.shape
        logits = x.view(batch_size * seq_length, vocab_size)
        targets = y.view(batch_size * seq_length)
        loss = self.loss_fn(logits, targets)

        return loss, []
