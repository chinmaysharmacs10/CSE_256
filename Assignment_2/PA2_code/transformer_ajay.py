import torch
import math
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.query_key_value_layer = nn.Linear(n_embd, 3 * n_embd)
        self.linear_layer = nn.Linear(n_embd, n_embd)

    @staticmethod
    def scaled_dot_product(query, key, value, mask=None):
        d_k = query.size()[-1]
        scaled_query_key_product = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scaled_query_key_product += mask
        attention = F.softmax(scaled_query_key_product, dim=-1)
        values = torch.matmul(attention, value)
        return values, attention

    def forward(self, x, mask=None):
        batch_size, sequence_length, n_embd = x.shape
        query_key_value = self.query_key_value_layer(x)

        # break query_key_value to num_heads
        query_key_value = query_key_value.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        query_key_value = query_key_value.permute(0, 2, 1, 3)
        query, key, value = query_key_value.chunk(3, dim=-1)    # chunk based on last dimension
        values, attention_weights = self.scaled_dot_product(query, key, value, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out, attention_weights


class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, epsilon=1e-5):
        super().__init__()
        self.parameter_shape = parameter_shape
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))

    def forward(self, inputs):  # inputs shape = batch_size x sequence_length x embedding_size
        # last dimension along which we want to perform layer normalization
        dims = [-(i + 1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)  # shape = batch_size x sequence_length x 1
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)   # shape = batch_size x sequence_length x 1
        std = (var + self.epsilon).sqrt()   # shape = batch_size x sequence_length x 1
        y = (inputs - mean) / std   # shape = batch_size x sequence_length x embedding_shape
        out = (self.gamma * y) + self.beta
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear_layer_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_layer_2 = nn.Linear(4 * n_embd, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_embd, num_heads)
        # self.layer_norm_1 = LayerNormalization([n_embd])
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.dropout_1 = nn.Dropout(drop_prob)
        self.feed_forward = FeedForward(n_embd, drop_prob)
        # self.layer_norm_2 = LayerNormalization([n_embd])
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.dropout_2 = nn.Dropout(drop_prob)

    def forward(self, x):
        residual_x = x
        x, attention_weights = self.multi_head_attention(x, mask=None)
        x = self.dropout_1(x)
        x = self.layer_norm_1(x + residual_x)
        residual_x = x
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = self.layer_norm_2(x + residual_x)
        return x, attention_weights


class Encoder(nn.Module):
    def __init__(self, n_embd, num_heads, drop_prob, num_layers, vocab_size, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x):
        token_embeddings = self.token_embedding_table(x)
        positional_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=device))
        x = token_embeddings + positional_embeddings

        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(x)
            attention_weights_list.append(attention_weights)

        return x, attention_weights_list
