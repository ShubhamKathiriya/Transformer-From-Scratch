import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Translation_Dataset
from tqdm import tqdm

import config
from tokenization import BPE_Tokenizer
from transformer import Transformer
from utils import calculate_bleu, plot_metrics, validate

config = config.config()


def train_epoch(model, dataloader, optimizer, criterion, device):

    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for eng, eng_mask, fr_1, fr_mask_1, fr_2, fr_mask_2 in tqdm(dataloader, desc="Training"):
        
        # fr_1 for teacher forcing
        # fr_2 for loss calculation

        eng, eng_mask, fr_1, fr_mask_1, fr_2, fr_mask_2 = eng.to(device), eng_mask.to(device), fr_1.to(device), fr_mask_1.to(device), fr_2.to(device), fr_mask_2.to(device)
        optimizer.zero_grad()

        output = model(eng, eng_mask, fr_1, fr_mask_1)
        
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), fr_2.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        pred = output.argmax(2)

        all_predictions.extend(pred.cpu().tolist())
        all_targets.extend(fr_2.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    bleu = calculate_bleu(all_predictions, all_targets)
    
    return avg_loss, bleu







def train(model, train_loader, val_loader, optimizer, criterion):

    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    train_bleus = []
    val_bleus = []
    
    for epoch in range(config.num_epochs):
        train_loss, train_bleu = train_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, val_bleu = validate(model, val_loader, criterion, config.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_bleus.append(train_bleu)
        val_bleus.append(val_bleu)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train BLEU: {train_bleu:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved!")
        
        print()

    plot_metrics(train_losses, val_losses, train_bleus, val_bleus)




def main():
    
    eng_tokenizer = BPE_Tokenizer(config.src_lang_vocab_size)
    fr_tokenizer = BPE_Tokenizer(config.tgt_lang_vocab_size)

    eng_tokenizer.train_tokenizer(config.train_eng)
    fr_tokenizer.train_tokenizer(config.train_fr)


    train_eng_sentence, train_eng_attention_mask = eng_tokenizer.tokenization(config.train_eng , "train")
    train_fr_sentence, train_fr_attention_mask =fr_tokenizer.tokenization(config.train_fr , "train")

    dev_eng_sentence, dev_eng_attention_mask = eng_tokenizer.tokenization(config.dev_eng, "dev")
    dev_fr_sentence, dev_fr_attention_mask = fr_tokenizer.tokenization(config.dev_fr , "dev")


    print(f"train english sentence:  {len(train_eng_sentence)}")
    print(f"train french sentence:  {len(train_fr_sentence)}")
    print(f"dev english sentence:  {len(dev_eng_sentence)}")
    print(f"dev french sentence:  {len(dev_fr_sentence)}")

    train_dataset = Translation_Dataset(train_eng_sentence , train_fr_sentence , train_eng_attention_mask , train_fr_attention_mask)
    dev_dataset = Translation_Dataset(dev_eng_sentence , dev_fr_sentence, dev_eng_attention_mask, dev_fr_attention_mask)

    print(f'english vocab size:- {eng_tokenizer.vocab_size()}   ||    french vocab size:-  {fr_tokenizer.vocab_size()}')

    train_loader = DataLoader(train_dataset , batch_size=config.batch_size , shuffle=True)
    dev_loader = DataLoader(dev_dataset , batch_size=config.batch_size , shuffle=False)

    print(f'train loader :- {len(train_loader)}    ||   dev dataloder :- {len(dev_loader)}')
    
    
    model = Transformer(eng_tokenizer , fr_tokenizer).to(config.device)
    optimizer = optim.AdamW(model.parameters() , lr = config.learning_rate , weight_decay = config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=eng_tokenizer.token_to_id(config.PAD_TOKEN))

    train(model , train_loader , dev_loader , optimizer , criterion)

    eng_tokenizer.save_tokenizer(config.eng_tokenizer_path)
    fr_tokenizer.save_tokenizer(config.fr_tokenizer_path)



if __name__ == "__main__":
    main()


