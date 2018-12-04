import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(CNNTextModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.conv13 = nn.Conv2d(1, 100, kernel_size=(3, embedding_dim))
        self.conv14 = nn.Conv2d(1, 100, kernel_size=(4, embedding_dim))
        self.conv15 = nn.Conv2d(1, 100, kernel_size=(5, embedding_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def _conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, 100, L-i+1)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # (N, 100)
        return x

    def forward(self, x):
        """
        Forward function.
        Arguments:
            x: Tensor of shape (batch_size, max_len)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        x = self.embed(x)  # (N, L, D)

        x = x.unsqueeze(1)  # (N, 1, L, D)

        x1 = self._conv_and_pool(x, self.conv13)  # (N, 100)
        x2 = self._conv_and_pool(x, self.conv14)  # (N, 100)
        x3 = self._conv_and_pool(x, self.conv15)  # (N, 100)

        x = torch.cat((x1, x2, x3), dim=1)  # (N, 300)
        x = self.dropout(x)
        logits = self.fc(x)  # (N, C)

        return logits
