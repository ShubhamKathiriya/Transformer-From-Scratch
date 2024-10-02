import torch 

class config:
    train_eng = '../ted-talks-corpus/train.en'
    train_fr = '../ted-talks-corpus/train.fr'
    test_eng = '../ted-talks-corpus/test.en'
    test_fr = '../ted-talks-corpus/test.fr'
    dev_eng = '../ted-talks-corpus/dev.en'
    dev_fr = '../ted-talks-corpus/dev.fr'

    best_model_path = '../best_transformer.pth'
    eng_tokenizer_path = '../english_tokenizer.txt'
    fr_tokenizer_path = '../french_tokenizer.txt'
    output_file_path = '../output_file.txt'

    
    # transformer parameters
    num_encoder_layers = 2
    num_decoder_layers = 2
    num_heads = 4

    # model parameters
    embedding_dim = 128
    model_dim = 128
    pe_dim = 128
    ff_dim = 512
    
    max_len = 25  # excluding <sos> and <eos>
    total_len = max_len + 2 # including <sos> and <eos>

    START_TOKEN = '<SOS>'
    END_TOKEN = '<EOS>'
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    src_lang_vocab_size = 30000
    tgt_lang_vocab_size = 30000


    # training parameters
    dropout = 0.2
    num_epochs = 10
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    