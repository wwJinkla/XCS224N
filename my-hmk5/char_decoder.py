#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        
        super(CharDecoder, self).__init__()
        self.vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, self.vocab_size, bias=True)
        self.decoderCharEmb = nn.Embedding(self.vocab_size, char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab 

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

        char_embed = self.decoderCharEmb(input) # shape(length, batch, char_embedding_size)
        h_ts, dec_hidden = self.charDecoder(char_embed, dec_hidden) # h_ts shape (legnth, batch, hidden_size)
        scores = self.char_output_projection(h_ts) # shape (length, batch, vocab_size)

        return scores, dec_hidden
        
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
        
        input = char_sequence[:-1] # shape (ll, batch) where ll = length - 1
        target = char_sequence[1:]  # shape (ll, batch)

        scores, _ = self.forward(input, dec_hidden) # (ll, batch, self.vocab_size)
        
        # reshape scores (ll, batch, self.vocab_size) -> (ll * batch, self.vocab_size). NOTE: I found this oddly strange...
        scores = scores.view(-1, scores.shape[2])
        # reshape target (ll, batch) -> (ll * batch)
        target = target.contiguous().view(-1)
        # use nn.CrossEntropyLoss. To make sure padding characters do not contribute, use ignore_index. Set reduction to sum
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'],
                                   reduction='sum')

        return loss(scores, target)
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

        batch_size = initialStates[0].shape[1]
        current_char_index = torch.tensor([[self.target_vocab.start_of_word] * batch_size], device=device) # shape (1, batch_size)
        decodedWords = []
        dec_hidden = initialStates

        # Compute, in batches, the predicted characters.
        # Store, in batches, the predicted character for each character position, up to max_length:
        #                    [[charIdx_00, charIdx_01,..., charIdx_0m],
        #                     [charIdx_10, charIdx_11,..., charIdx_1m],
        #                                          ...
        #                     [charIdx_b0, charIdx_b1,..., charIdx_bm]]
        # where each row represents a word sequence. 
        current_words = []

        for t in range(max_length):
            scores, dec_hidden = self.forward(current_char_index, dec_hidden) # scores shape (1, batch_size, hidden_size)
            p = nn.functional.softmax(scores, dim=2)
            current_char_index = p.argmax(dim=2) # shape (1, batch_size)
            current_words.append(current_char_index.squeeze(0).tolist())
        
        # Now truncate by iterating thru each word 
        # NOTE: think of a more clever way of doing this
        current_words = torch.tensor(current_words, device=device).transpose(1,0) # NOTE: tensor.T is not backward-supported
        for word in current_words:
            decoded = ''
            for char_idx in word.tolist():
                if char_idx == self.target_vocab.end_of_word:
                    break
                else:
                    decoded += self.target_vocab.id2char[char_idx]
            decodedWords.append(decoded)

        return decodedWords
        
        ### END YOUR CODE

