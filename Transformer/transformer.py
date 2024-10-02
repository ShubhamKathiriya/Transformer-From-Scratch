import torch
import torch.nn as nn

import config
from encoder import Encoder
from decoder import Decoder

config = config.config()


class Transformer(nn.Module):

    def __init__(self, src_tokenizer , tgt_tokenizer):
        super().__init__()

        self.src_vocab_size = src_tokenizer.vocab_size()
        self.tgt_vocab_size = tgt_tokenizer.vocab_size()
        
        self.encoder = Encoder(self.src_vocab_size)
        self.decoder = Decoder(self.tgt_vocab_size)
        
        self.src_pad_index = src_tokenizer.token_to_id(config.PAD_TOKEN) 
        self.tgt_pad_index = tgt_tokenizer.token_to_id(config.PAD_TOKEN) 

        self.dropout = nn.Dropout(config.dropout)


    def forward(self, src, src_mask , tgt , tgt_mask):
    
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return output
    
    