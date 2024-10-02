import torch.nn as nn
import config
from utils import MultiheadAttention, FeedForward, PositionalEncoding

config = config.config()


class EncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention = MultiheadAttention()
        self.feed_forward = FeedForward()

        self.norm = nn.LayerNorm(config.model_dim)

        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)  
        x = self.norm(x) 
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm(x)  
        
        return x



class Encoder(nn.Module):

    def __init__(self , vocab_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size , config.embedding_dim)
        self.position = PositionalEncoding(config.embedding_dim, 0)

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.num_encoder_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x, mask=None):


        x = self.embedding(x) * (config.embedding_dim ** 0.5)
        x = self.position(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)