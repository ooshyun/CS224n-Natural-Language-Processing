#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super().__init__()

        # padding method associate between vocab and Embedding method
        pad_token_idx = target_vocab.char2id['<pad>']

        self.char_decoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id), bias=True)
        self.decoder_char_embed = nn.Embedding(len(target_vocab.char2id), char_embedding_size, pad_token_idx)
        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, char_sequence, dec_hidden=None):
        """ Forward pass of character decoder.

        @param char_sequence: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_char_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b

        x_embed = self.decoder_char_embed.forward(char_sequence)
        hiddens, dec_char_hidden = self.char_decoder.forward(x_embed, dec_hidden)
        scores = self.char_output_projection.forward(hiddens)
        return scores, dec_char_hidden

        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        input_char_sequence, target_char_sequence = char_sequence[:-1], char_sequence[1:]

        scores, dec_char_hidden = self.forward(input_char_sequence, dec_hidden)
        flat_scores = scores.view(-1, scores.shape[-1])
        flat_scores = torch.flatten(scores, start_dim=0, end_dim=1)
        flat_targets = torch.flatten(target_char_sequence)

        cross_entropy = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        loss = cross_entropy.forward(flat_scores, flat_targets)

        return loss

        ### END YOUR CODE

    def decode_greedy(self, initial_states, device, max_length=21):
        """ Greedy decoding
        @param initial_states: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decoded_words: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = initial_states[0].shape[1]

        decoded_words = [""] * batch_size
        decoded_done = [False] * batch_size

        dec_hidden = initial_states

        # 1 batch -> <unk> >= 2 -> How?
        input_char = torch.tensor([[self.target_vocab.start_of_word] * batch_size], device=device)

        for _ in range(max_length):
            scores, dec_hidden = self.forward(input_char, dec_hidden)
            input_char = scores.argmax(dim=2)

            for idx, char_id in enumerate(input_char.detach().squeeze(0)):
                if decoded_done[idx] is True:
                    continue
                if char_id == self.target_vocab.end_of_word:
                    decoded_done[idx] = True
                    if all(decoded_done):
                        break
                    continue
                char = self.target_vocab.id2char[int(char_id)]
                decoded_words[idx] += char

        return decoded_words

        ### END YOUR CODE

