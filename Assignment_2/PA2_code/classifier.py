import torch
from torch import nn
from transformer import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, block_size, vocab_size, n_input, n_hidden, n_output, drop_prob,
                 batch_norm):
        super(Classifier, self).__init__()
        self.encoder = Encoder(n_embd=n_embd, num_heads=n_head, num_layers=n_layer,
                               vocab_size=vocab_size, block_size=block_size, drop_prob=drop_prob)
        self.fc_1 = nn.Linear(n_input, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=drop_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        encoder_output, attention_weights = self.encoder(x)
        pooled_output = torch.mean(encoder_output, dim=1)
        # out = self.fc_2(self.relu(self.bn(self.fc_1(pooled_output))))
        out = self.fc_1(pooled_output)
        if self.batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        return out, attention_weights
