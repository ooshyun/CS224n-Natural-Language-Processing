#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import math
import unittest
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, char_embed_size, word_embed_size, window_size):
        super().__init__()
        self.conv1d = nn.Conv1d(char_embed_size, word_embed_size, kernel_size=window_size, bias=True)

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        x_cnv = self.conv1d.forward(x_emb)
        x_act = torch.relu(x_cnv)
        x_conv_out, x_conv_idx = x_act.max(dim=2)
        return x_conv_out


class CNNSanityChecks(unittest.TestCase):

    def test_shape(self):
        max_word_len = 15
        batch_size, char_embed_size, word_embed_size, window_size = 64, 20, 80, 5
        cnn = CNN(char_embed_size, word_embed_size, window_size)

        x_emb = torch.randn([batch_size, char_embed_size, max_word_len])
        x_conv_out = cnn.forward(x_emb)

        self.assertEqual(x_conv_out.shape, (batch_size, word_embed_size))


### END YOUR CODE

