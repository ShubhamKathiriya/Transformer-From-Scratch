import torch.nn as nn

import config 
from utils import MultiheadAttention, FeedForward, PositionalEncoding

config = config.config()



class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attention = MultiheadAttention()
        self.cross_attention = MultiheadAttention()
        self.ffnn = FeedForward()

        self.norm = nn.LayerNorm(config.model_dim)        
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x, encoder_output, src_mask, target_mask):
        
        # Self Attention
        self_atten_output = self.self_attention(x, x, x, target_mask)
        x = self.norm(x + self.dropout(self_atten_output))

        # Cross Attention
        cross_atten_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm(x + self.dropout(cross_atten_output))

        # Feed Forward
        ffnn_output = self.ffnn(x)
        x = self.norm(x + self.dropout(ffnn_output))

        return x
    


class Decoder(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size , config.embedding_dim)
        self.position = PositionalEncoding(config.embedding_dim, 0)

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.num_decoder_layers)])
        self.fc_out = nn.Linear(config.model_dim, vocab_size)
        self.norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, src_mask, target_mask):

        x = self.embedding(x) * (config.embedding_dim ** 0.5)
        
        x = self.position(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        
        x = self.fc_out(x)
        return x