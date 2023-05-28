import torch
import torch.nn as nn

import layers
from layers import masked_softmax


def create_model(name, word_vectors, hidden_size, drop_prob=0.0):
    if name == "lstm":
        return LSTM_QA_Model(
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            bidirectional=False,
            drop_prob=drop_prob,
        )
    elif name == "bilstm":
        return LSTM_QA_Model(
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            bidirectional=True,
            drop_prob=drop_prob,
        )
    elif name == "lstm_attention":
        return LSTM_Attention_QA_Model(
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            bidirectional=False,
            drop_prob=drop_prob,
        )
    elif name == "bilstm_attention":
        return LSTM_Attention_QA_Model(
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            bidirectional=True,
            drop_prob=drop_prob,
        )
    else:
        raise NotImplementedError(f"Unknown model: {name}")


class LSTM_QA_Model(nn.Module):
    def __init__(self, word_vectors, hidden_size, bidirectional, drop_prob=0.0):
        super().__init__()

        self.drop_prob = drop_prob

        self.emb = layers.Embedding(
            word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob
        )

        self.enc = layers.RNNEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            drop_prob=drop_prob,
        )

        self.fc = nn.Linear(
            in_features=2 * hidden_size if bidirectional else hidden_size,
            out_features=hidden_size,
        )

        self.logits_1 = nn.Linear(hidden_size, 1)
        self.logits_2 = nn.Linear(hidden_size, 1)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)
        q_emb = self.emb(qw_idxs)

        _, q_state = self.enc(q_emb, q_len)
        c_enc, _ = self.enc(c_emb, c_len, q_state)

        c_enc = self.fc(c_enc)

        logits_1 = self.logits_1(c_enc)
        logits_2 = self.logits_2(c_enc)

        log_p1 = masked_softmax(logits_1.squeeze(), c_mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), c_mask, log_softmax=True)

        return log_p1, log_p2


class LSTM_Attention_QA_Model(nn.Module):
    def __init__(self, word_vectors, hidden_size, bidirectional, drop_prob=0.0):
        super().__init__()
        self.emb = layers.Embedding(
            word_vectors=word_vectors, hidden_size=hidden_size, drop_prob=drop_prob
        )

        self.enc = layers.RNNEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            drop_prob=drop_prob,
        )

        self.att = layers.Attention(
            hidden_size=hidden_size * 2 if bidirectional else hidden_size,
            drop_prob=drop_prob,
        )

        self.mod = layers.RNNEncoder(
            input_size=8 * hidden_size if bidirectional else 4 * hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            drop_prob=drop_prob,
        )

        self.out = layers.AttentionOutput(
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            drop_prob=drop_prob,
        )

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)
        q_emb = self.emb(qw_idxs)

        c_enc, _ = self.enc(c_emb, c_len)
        q_enc, _ = self.enc(q_emb, q_len)

        att = self.att(c_enc, q_enc, c_mask, q_mask)

        mod, _ = self.mod(att, c_len)

        out = self.out(att, mod, c_mask)

        return out
