#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self, embed_size,channel_size, kernel_size):
        """
        Convolutional Neural Network Init

        @param embed_size : int, embedding size
        @param channel_size: int, filter channel , channel_size = embed_size
        @param kernel_size: int, kernal size
        """

        super(CNN,self).__init__()

        self.embed_size = embed_size
        self.conv = nn.Conv1d(embed_size,channel_size, kernel_size)
    
    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        input: reshaped char embedding batch
        output: cnn output tensor

        @param x_reshaped : torch.Tensor, (batch_size, embed_size, m_word)

        """
        x_conv = self.conv(x_reshaped)
        x_conv_out = F.max_pool1d(F.relu(x_conv),x_conv.shape[2])
        x_conv_out = torch.squeeze(x_conv_out, dim=2)
        return x_conv_out



### END YOUR CODE

