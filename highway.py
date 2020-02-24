#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class Highway(torch.nn.Module):
    def __init__(self, embed_size,dropout_rate):
        """
        In the constructor we instantiate Highway Network modules and assign them as
        member variables.
        @param embed_size: int, Embedding size
        @param dropout_rate: float, dropout rate
        
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size 
        self.dropout_rate = dropout_rate

        self.projection = nn.Linear(embed_size,embed_size, bias=True)
        self.gate = nn.Linear(embed_size,embed_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        input: the sentences bach
        output: produce the word embedding of max word length 

        @param x_conv_out : torch.Tensor, (batch_size,embed_size)
        
        return word_embedding: torch.Tensor,(max_sentence_length, batch_size, embed_size)
        """
        x_proj = F.relu(self.projection(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x_highway = x_gate*x_proj + (1.0-x_gate)*x_conv_out

        word_embedding = self.dropout(x_highway)

        return word_embedding
        

### END YOUR CODE 

