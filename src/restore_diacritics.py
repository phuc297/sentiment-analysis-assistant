import sacrebleu
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import math
import string
import re
import pickle
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# --- TRANSFORMER MODULES ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V, mask=None):
        Q, K, V = Q.float(), K.float(), V.float()
        scores = Q.matmul(K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(scores, dim=-1)
        return attn.matmul(V)

    def split_heads(self, x):
        b, seq_len, _ = x.size()
        return x.view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        b, h, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        out = self.scaled_dot_product(Q, K, V, mask)
        out = self.combine_heads(out)
        return self.W_o(out)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(
            x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8, num_layers=2, d_ff=512, max_len=70, dropout=0.1):
        super().__init__()
        self.enc_emb = nn.Embedding(src_vocab_size, d_model)
        self.dec_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak = torch.tril(torch.ones(
            1, 1, seq_len, seq_len, device=tgt.device)).bool()
        tgt_mask = tgt_mask & nopeak
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        x = self.dropout(self.pos_enc(self.enc_emb(src)))
        y = self.dropout(self.pos_enc(self.dec_emb(tgt)))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        for layer in self.dec_layers:
            y = layer(y, x, src_mask, tgt_mask)
        return self.fc_out(y)


# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- CLEAN TEXT ---
def clean_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_cleaning(df):
    df['no_diacritics_clean'] = df['no_diacritics'].astype(
        str).apply(clean_text)
    df['with_diacritics_clean'] = df['with_diacritics'].astype(
        str).apply(clean_text)
    df['with_diacritics_clean'] = df['with_diacritics_clean'].apply(
        lambda x: '<start> ' + x + ' <end>')
    return df


class RestoreDiacriticsModel():
    def __init__(self, model_save_path):
        with open(os.path.join(model_save_path, "src_tokenizer.pkl"), "rb") as f:
            self.src_tokenizer = pickle.load(f)

        with open(os.path.join(model_save_path, "tgt_tokenizer.pkl"), "rb") as f:
            self.tgt_tokenizer = pickle.load(f)

        model_path = os.path.join(model_save_path, "transformer_model.pt")
        self.model = Transformer(len(self.src_tokenizer.word_index)+1,
                                 len(self.tgt_tokenizer.word_index)+1)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)

    def greedy_decode(self, input_sentence, max_len=70):
        self.model.eval()
        idx2word = {idx: word for word,
                    idx in self.tgt_tokenizer.word_index.items()}
        idx2word[0] = '<pad>'

        cleaned = clean_text(input_sentence)
        seq = pad_sequences(self.src_tokenizer.texts_to_sequences(
            [cleaned]), maxlen=max_len, padding='post')
        src_tensor = torch.LongTensor(seq).to(device)

        start_token = self.tgt_tokenizer.word_index['<start>']
        end_token = self.tgt_tokenizer.word_index['<end>']
        tgt_tokens = [start_token]
        decoded = []

        for _ in range(max_len):
            tgt_tensor = torch.LongTensor([tgt_tokens]).to(device)
            with torch.no_grad():
                out = self.model(src_tensor, tgt_tensor)
                logits = out[0, -1, :]
                next_token = torch.argmax(logits).item()

            if next_token == end_token:
                break
            decoded.append(idx2word.get(next_token, '<unk>'))
            tgt_tokens.append(next_token)

        return ' '.join(decoded)


if __name__ == "__main__":

    import config
    # --- TEST PREDICTIONS ---
    test_sentences = [
        "hom nay buon qua",
        "tuyet voi qua",
        "hom nay that la thu vi"
    ]

    restore_diacritics = RestoreDiacriticsModel(config.MODEL_SAVE_PATH)

    print("Kết quả dự đoán:")
    for sent in test_sentences:
        print(
            f"Input: {sent}\nOutput: {restore_diacritics.greedy_decode(sent)}\n")
