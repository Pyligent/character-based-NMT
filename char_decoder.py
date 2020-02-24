#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder,self).__init__()

        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size
        self.target_vocab = target_vocab
        self.padding_idx = self.target_vocab.char2id['<pad>']

        # init self.charDecoder
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)

        # init self.char_output_projection
        self.char_output_projection = nn.Linear(hidden_size,len(self.target_vocab.char2id))

        # init self.decoderCharEmb
        num_embeddings = len(self.target_vocab.char2id)
        
        self.decoderCharEmb = nn.Embedding(num_embeddings,char_embedding_size,self.padding_idx)

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # input shape: (length, batch), output shape: (length, batch, embedding_size)
        char_embedding = self.decoderCharEmb(input)
        # input to LSTM, output shape: (length, batch, hidden_size)
        hidden_s, dec_hidden = self.charDecoder(char_embedding,dec_hidden)
        # scores shape : (length, batch, self.vocab_size)
        scores = self.char_output_projection(hidden_s)

        return scores,dec_hidden

        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # input char_sequence into forward , source remove the <end> token and target remove <begin> token
        source_char_seq = char_sequence[:-1]
        target_char_seq = char_sequence[1:]

        scores, dec_hidden = self.forward(source_char_seq , dec_hidden)

        loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx, reduction='sum')
        
        # scores shape : (length, batch, self.vocab_size)
        # CrossEntropyLoss input shape (minibatch, C), target is as C
        cross_entropy_loss = loss(scores.permute(1,2,0), torch.t(target_char_seq))

        return cross_entropy_loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        output_word =[]
        start_of_word_idx = self.target_vocab.start_of_word
        end_of_word_idx = self.target_vocab.end_of_word

        dec_hidden = initialStates
        # dec_hidden: (length, batch, hidden_size)
        batch_size = dec_hidden[0].shape[1]

        # current char <- <start>
        current_char = torch.tensor([[self.target_vocab.start_of_word]*batch_size], device = device)

        output = [[-1] * max_length for _ in range(batch_size)]
        for i in range(max_length):
            # (1, batch_size, char_embed_size)
            input_char_embedding = self.decoderCharEmb(current_char)
           
            # dec_hidden: a 2-tuple of Tensors of (1, batch, hidden_size)
            _, dec_hidden = self.charDecoder(input_char_embedding, dec_hidden)

            # (batch_size, char_vocab_size)
            h_t = torch.squeeze(dec_hidden[0], dim=0)
            c_t = self.char_output_projection(h_t)
            # compute the probabilities
            p_t = F.softmax(c_t, dim=1)
            
            # most likely next char
            next_char_idx = torch.argmax(p_t, dim=1)
            # (1, batch_size)
            current_char = torch.unsqueeze(next_char_idx, dim=0)

            idx_list = next_char_idx.tolist()
            for j in range(batch_size):
                output[j][i] = idx_list[j]

        output = [''.join([self.target_vocab.id2char[k]
                           for k in next_char_idx]) for next_char_idx in output]
        
        def split_end(w):
            return w.split(self.target_vocab.id2char[end_of_word_idx])[0]
        decodedWords = list(map(split_end,output))
        
        return decodedWords
        
        ### END YOUR CODE

