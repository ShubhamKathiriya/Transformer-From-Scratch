import torch
import string
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import config


config = config()


class BPE_Tokenizer:
    
    def __init__(self, vocab_size=None):
        self.tokenizer = HFTokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        
        if vocab_size is not None:
            self.trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=[
                    config.PAD_TOKEN,
                    config.START_TOKEN,
                    config.END_TOKEN,
                    config.UNK_TOKEN
                ]
            )
        
    def train_tokenizer(self, input_files):
        if not isinstance(input_files, str):
            raise ValueError("input_files should be a list of file paths.")
        
        self.tokenizer.train(files=[input_files], trainer=self.trainer)

    def save_tokenizer(self, path="bpe_tokenizer.json"):
        self.tokenizer.save(path)

    def load_tokenizer(self, path="bpe_tokenizer.json"):
        """Load the tokenizer from a JSON file."""
        self.tokenizer = HFTokenizer.from_file(path)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def tokenization(self, file, mode):
        tokenized_sentences = []
        attention_masks = []

        # Create a translation table for removing punctuation
        translator = str.maketrans('', '', string.punctuation)

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                # Convert to lowercase and remove punctuation
                line = line.lower().translate(translator)
                encoded = self.tokenizer.encode(line.strip())
                token_ids = encoded.ids

                if mode == "train":
                    token_ids += [self.tokenizer.token_to_id(config.END_TOKEN)]
                else:
                    token_ids = [self.tokenizer.token_to_id(config.START_TOKEN)] + token_ids + [self.tokenizer.token_to_id(config.END_TOKEN)]

                attention_mask = [1] * len(token_ids)

                if len(token_ids) < config.total_len:
                    pad_length = config.total_len - len(token_ids)
                    token_ids += [self.tokenizer.token_to_id(config.PAD_TOKEN)] * pad_length
                    attention_mask += [0] * pad_length
                else:
                    token_ids = token_ids[:config.total_len - 1] + [self.tokenizer.token_to_id(config.END_TOKEN)]
                    attention_mask = attention_mask[:config.total_len]

                tokenized_sentences.append(token_ids)
                attention_masks.append(attention_mask)

        return tokenized_sentences, attention_masks
