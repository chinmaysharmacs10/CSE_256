import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, drop_prob):
        super(AttentionHead, self).__init__()
        self.d_k = head_size
        self.W_q = nn.Linear(n_embd, head_size, bias=False)
        self.W_k = nn.Linear(n_embd, head_size, bias=False)
        self.W_v = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, mask=None):
        batch_size, seq_length, n_embd = x.shape
        query, key, value = self.W_q(x), self.W_k(x), self.W_v(x)
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(self.tril[:seq_length, :seq_length] == 0, float('-inf'))

        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = torch.matmul(attention_weights, value)
        return out, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, drop_prob):
        super(MultiHeadAttention, self).__init__()
        head_size = n_embd // num_heads
        self.attention_heads = nn.ModuleList([AttentionHead(n_embd, head_size, block_size, drop_prob)
                                              for _ in range(num_heads)])
        self.W_o = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, mask=None):
        outputs = []
        attention_weights = []
        for attention_head in self.attention_heads:
            out, attention_weight = attention_head(x, mask)
            outputs.append(out)
            attention_weights.append(attention_weight)
        out = torch.cat(outputs, dim=-1)
        out = self.W_o(out)
        out = self.dropout(out)
        # attention_weights = torch.stack(attention_weights, dim=0).transpose(0, 1)
        return out, attention_weights


class FeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim, drop_prob):
        super(FeedForward, self).__init__()
        self.fc_1 = nn.Linear(n_embd, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, ff_dim, block_size, drop_prob):
        super(Block, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_embd, num_heads, block_size, drop_prob)
        self.feed_forward = FeedForward(n_embd, ff_dim, drop_prob)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.dropout_1 = nn.Dropout(p=drop_prob)
        self.dropout_2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask=None):
        residual_x = x
        x, attention_weights = self.multi_head_attention(x, mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(x + residual_x)
        residual_x = x
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = self.layer_norm_2(x + residual_x)
        return x, attention_weights


class Encoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size, drop_prob):
        super(Encoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.encoder_layers = nn.ModuleList([Block(n_embd=n_embd, num_heads=num_heads, ff_dim=(4 * n_embd),
                                                   block_size=block_size, drop_prob=drop_prob) for _ in range(num_layers)])

    def forward(self, x):
        token_embeddings = self.token_embedding_table(x)
        positional_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=device))
        x = token_embeddings + positional_embeddings
        attention_weights_list = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x)
            attention_weights_list.extend(attention_weights)
        return x, attention_weights_list


class Decoder(nn.Module):
    def __init__(self, n_embd, num_heads, num_layers, vocab_size, block_size, ff_dim):
        super(Decoder, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.decoder_layers = nn.ModuleList([Block(n_embd=n_embd, num_heads=num_heads, ff_dim=ff_dim,
                                                   block_size=block_size, drop_prob=0.0) for _ in range(num_layers)])
        self.fc = nn.Linear(n_embd, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        token_embeddings = self.token_embedding_table(x)
        positional_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=device))
        x = token_embeddings + positional_embeddings

        attention_weights_list = []
        for layer in self.decoder_layers:
            x, attention_weights = layer(x, mask="mask")
            attention_weights_list.extend(attention_weights)

        x = self.fc(x)

        batch_size, seq_length, vocab_size = x.shape
        logits = x.view(batch_size * seq_length, vocab_size)
        targets = y.view(batch_size * seq_length)
        loss = self.loss_fn(logits, targets)

        return loss, attention_weights_list
