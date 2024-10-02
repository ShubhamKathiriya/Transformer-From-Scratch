import matplotlib.pyplot as plt
import math
from sacrebleu.metrics import BLEU
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

import config

config = config.config()



class HeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, model_dim):
        super().__init__()
        
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(embedding_dim, self.head_dim, bias=False)

    def forward(self, q, k, v, mask=None):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * q.shape[-1]

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))


        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)  # Shape should be [batch_size, seq_len, head_dim]
        
        return attn_output
    


class MultiheadAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention(config.embedding_dim, config.num_heads, config.model_dim) for _ in range(config.num_heads)])
        self.projection = nn.Linear(config.model_dim, config.model_dim)  # Projection should match model_dim
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
    
        x = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)

        if x.shape[-1] != config.model_dim:
            print(f"Expected shape after concatenating heads: [batch_size, seq_len, {config.model_dim}], but got {x.shape}")
        

        x = self.dropout(self.projection(x))
        return x


class FeedForward(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.model_dim, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, config.model_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    



class Translation_Dataset(Dataset):
    
    def __init__(self, eng_tokens , fr_tokens, eng_attention_mask , fr_attention_mask):
        self.eng_tokens = eng_tokens
        self.fr_tokens = fr_tokens
        self.eng_attention_mask = eng_attention_mask
        self.fr_attention_mask = fr_attention_mask
    
    def __len__(self):
        return len(self.eng_tokens)

    def __getitem__(self, index):
        eng = torch.tensor(self.eng_tokens[index])
        eng_mask = torch.tensor(self.eng_attention_mask[index])
        fr_1 = torch.tensor(self.fr_tokens[index][:-1])   # teacher forcing 
        fr_mask_1 = torch.tensor(self.fr_attention_mask[index][:-1]) # teacher forcing 
        fr_2 = torch.tensor(self.fr_tokens[index][1:])  # loss purpose
        fr_mask_2 = torch.tensor(self.fr_attention_mask[index][1:])   # loss purpose

        return eng , eng_mask , fr_1 , fr_mask_1, fr_2 , fr_mask_2
    
    


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    


def calculate_bleu(predictions, targets):
    bleu = BLEU()
    
    predictions = [' '.join(map(str, pred)) for pred in predictions]
    targets = [[' '.join(map(str, tgt))] for tgt in targets]
    
    return bleu.corpus_score(predictions, targets).score



def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for eng, eng_mask, fr_1, fr_mask_1, fr_2, fr_mask_2 in tqdm(dataloader, desc="Validating"):
                    
            # fr_1 for teacher forcing
            # fr_2 for loss calculation
            eng, eng_mask, fr_1, fr_mask_1, fr_2, fr_mask_2 = eng.to(device), eng_mask.to(device), fr_1.to(device), fr_mask_1.to(device), fr_2.to(device), fr_mask_2.to(device)
    
            
            output = model(eng, eng_mask, fr_1, fr_mask_1)
            
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), fr_2.contiguous().view(-1))
            total_loss += loss.item()
            
            pred = output.argmax(2)

            all_predictions.extend(pred.cpu().tolist())
            all_targets.extend(fr_2.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    bleu = calculate_bleu(all_predictions, all_targets)
    
    return avg_loss, bleu



def plot_metrics(train_losses, val_losses, train_bleus, val_bleus):
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots for loss and BLEU scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Losses
    ax1.plot(epochs, train_losses, 'b', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot BLEU scores
    ax2.plot(epochs, train_bleus, 'b', label='Training BLEU')
    ax2.plot(epochs, val_bleus, 'r', label='Validation BLEU')
    ax2.set_title('Training and Validation BLEU Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('BLEU Score')
    ax2.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


