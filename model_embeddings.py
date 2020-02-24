#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        
        ## Create embedding
        pad_token_idx = vocab.char2id['<pad>']

        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, padding_idx=pad_token_idx)
        
        ## Input to 1D Conv. Net
        self.cnn = CNN(embed_size, embed_size, kernel_size = 5)

        ## Highway Network
        self.highway = Highway(embed_size, dropout_rate = 0.3)

        ### END YOUR CODE

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f

        
        sents_embedding = self.embeddings(input)  
        # (sentence_length, batch_size, max_word_length, embed_size)

        sents_reshaped = torch.transpose(sents_embedding,2,3)
        # (sentence_length, batch_size, embed_size, max_word_length)

        output = torch.zeros(sents_reshaped.shape[:3], device=input.device)
       
        for i, x_reshaped in enumerate(torch.split(sents_reshaped, 1, dim=0)):
            x_reshaped = torch.squeeze(x_reshaped, dim=0)  # (batch_size, embed_size, max_word_length)
            x_conv_out = self.cnn(x_reshaped)              # (batch_size, embed)
            
            x_embed = self.highway(x_conv_out)        # (batch_size, embed)
            output[i, :, :] = x_embed

        return output




        ### END YOUR CODE
