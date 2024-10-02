import torch
import torch.nn as nn
import os
from tqdm import tqdm

from tokenization import BPE_Tokenizer
from config import config
from utils import Translation_Dataset, validate, calculate_bleu
from torch.utils.data import DataLoader
from transformer import Transformer



def remove_pad(tokens, tokenizer):
    return [token for token in tokens if token != tokenizer.token_to_id(config.PAD_TOKEN)]



def test_and_write_sentences(model, dataloader, src_tokenizer, tgt_tokenizer, output_file):
    model.eval()
    all_predictions = []
    all_targets = []
    bleu_scores = []
    
    # Open file for writing
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with torch.no_grad():
            for eng, eng_mask, fr_1, fr_mask_1, fr_2, fr_mask_2 in tqdm(dataloader, desc="Testing"):
                eng, eng_mask, fr_1, fr_mask_1, fr_2, fr_mask_2 = (
                    eng.to(config.device),
                    eng_mask.to(config.device),
                    fr_1.to(config.device),
                    fr_mask_1.to(config.device),
                    fr_2.to(config.device),
                    fr_mask_2.to(config.device),
                )
                
                output = model(eng, eng_mask, fr_1, fr_mask_1)
                
                predictions = output.argmax(2)
                
                for pred_tokens, target_tokens in zip(predictions, fr_2):
                    pred_tokens_clean = remove_pad(pred_tokens.cpu().tolist(), tgt_tokenizer)
                    target_tokens_clean = remove_pad(target_tokens.cpu().tolist(), tgt_tokenizer)
                    
                    pred_sentence = tgt_tokenizer.decode(pred_tokens_clean)
                    target_sentence = tgt_tokenizer.decode(target_tokens_clean)
                    
                    bleu = calculate_bleu([pred_sentence], [target_sentence])
                    bleu_scores.append(bleu)
                    
                    f_out.write(f"{pred_sentence}\tBLEU: {bleu:.2f}\n")
                    
                    all_predictions.append(pred_sentence)
                    all_targets.append(target_sentence)

    print(f"Results written to {output_file}")


config = config()

eng_tokenizer = BPE_Tokenizer()
fr_tokenizer = BPE_Tokenizer()


eng_tokenizer.load_tokenizer(config.eng_tokenizer_path)
fr_tokenizer.load_tokenizer(config.fr_tokenizer_path)


test_eng_sentence, test_eng_attention_mask = eng_tokenizer.tokenization(config.test_eng , "test")
test_fr_sentence, test_fr_attention_mask =fr_tokenizer.tokenization(config.test_fr , "test")

print(f"test english sentence:  {len(test_eng_sentence)}")
print(f"test french sentence:  {len(test_fr_sentence)}")

test_dataset = Translation_Dataset(test_eng_sentence , test_fr_sentence , test_eng_attention_mask , test_fr_attention_mask)
test_loader = DataLoader(test_dataset , batch_size=config.batch_size , shuffle=False)

best_model = Transformer(eng_tokenizer , fr_tokenizer).to(config.device)
best_model.load_state_dict(torch.load(config.best_model_path))

criterion = nn.CrossEntropyLoss(ignore_index=eng_tokenizer.token_to_id(config.PAD_TOKEN))

validate(best_model, test_loader, criterion, config.device)
test_and_write_sentences(best_model, test_loader, eng_tokenizer, fr_tokenizer, config.output_file_path)
