from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dill
from janome.tokenizer import Tokenizer

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/', methods=['POST'])
@cross_origin(supports_credentials=True)
def index():
    inp_text = request.form['content']
    rep_text = reply(inp_text, j_tk, max_length=20)
    print("input:", inp_text)
    print("reply:", rep_text)

    return jsonify({"content": rep_text})


class Encoder(nn.Module):
    def __init__(self, n_h, n_vocab, n_emb, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()

        self.n_h = n_h
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding = nn.Embedding(n_vocab, n_emb)
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.gru = nn.GRU(
            input_size=n_emb,
            hidden_size=n_h,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        idx_pad = input_field.vocab.stoi["<pad>"]
        sentence_lengths = x.size()[1] - (x == idx_pad).sum(dim=1)

        y = self.embedding(x)
        y = self.embedding_dropout(y)
        y = nn.utils.rnn.pack_padded_sequence(
            y,
            sentence_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        y, h = self.gru(y)

        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        if self.bidirectional:
            y = y[:, :, :self.n_h] + y[:, :, self.n_h:]
            h = h[:self.num_layers] + h[self.num_layers:]
        return y, h


class Decoder(nn.Module):
    def __init__(self, n_h, n_out, n_vocab, n_emb, num_layers=1, dropout=0.0):
        super().__init__()

        self.n_h = n_h
        self.n_out = n_out
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(n_vocab, n_emb)
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.gru = nn.GRU(
            input_size=n_emb,
            hidden_size=n_h,
            batch_first=True,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(n_h*2, self.n_out)

    def forward(self, x, h_encoder, y_encoder):
        y = self.embedding(x)
        y = self.embedding_dropout(y)
        y, h = self.gru(y, h_encoder)

        y_tr = torch.transpose(y, 1, 2)
        ed_mat = torch.bmm(y_encoder, y_tr)
        attn_weight = F.softmax(ed_mat, dim=1)
        attn_weight_tr = torch.transpose(attn_weight, 1, 2)
        context = torch.bmm(attn_weight_tr, y_encoder)
        y = torch.cat([y, context], dim=2)

        y = self.fc(y)
        y = F.softmax(y, dim=2)

        return y, h


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, is_gpu=True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.is_gpu = is_gpu
        if self.is_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, x_encoder, x_decoder):
        if self.is_gpu:
            x_encoder, x_decoder = x_encoder.cuda(), x_decoder.cuda()

        batch_size = x_decoder.shape[0]
        n_time = x_decoder.shape[1]
        y_encoder, h = self.encoder(x_encoder)

        y_decoder = torch.zeros(batch_size, n_time, self.decoder.n_out)
        if self.is_gpu:
            y_decoder = y_decoder.cuda()

        for t in range(0, n_time):
            x = x_decoder[:, t:t+1]
            y, h = self.decoder(x, h, y_encoder)
            y_decoder[:, t:t+1, :] = y
        return y_decoder

    def predict(self, x_encoder, max_length=10):
        if self.is_gpu:
            x_encoder = x_encoder.cuda()

        batch_size = x_encoder.shape[0]
        n_time = max_length
        y_encoder, h = self.encoder(x_encoder)

        y_decoder = torch.zeros(batch_size, n_time, dtype=torch.long)
        if self.is_gpu:
            y_decoder = y_decoder.cuda()

        y = torch.ones(batch_size, 1, dtype=torch.long) * \
            input_field.vocab.stoi["<sos>"]
        for t in range(0, n_time):
            x = y
            if self.is_gpu:
                x = x.cuda()
            y, h = self.decoder(x, h, y_encoder)
            y = y.argmax(2)
            y_decoder[:, t:t+1] = y
        return y_decoder


input_field = torch.load("models/input.pkl", pickle_module=dill)
reply_field = torch.load("models/reply.pkl", pickle_module=dill)

is_gpu = False
n_h = 896
n_vocab_inp = len(input_field.vocab.itos)
n_vocab_rep = len(reply_field.vocab.itos)
n_emb = 300
n_out = n_vocab_rep
early_stop_patience = 5
num_layers = 1
bidirectional = True
dropout = 0.0
clip = 100

encoder = Encoder(n_h, n_vocab_inp, n_emb, num_layers,
                  bidirectional, dropout=dropout)
decoder = Decoder(n_h, n_out, n_vocab_rep, n_emb, num_layers, dropout=dropout)
seq2seq = Seq2Seq(encoder, decoder, is_gpu=is_gpu)

seq2seq.load_state_dict(torch.load(
    "models/model.pth", map_location=torch.device("cpu")))

j_tk = Tokenizer()


def reply(inp_text, tokenizer, max_length=10):
    words = [tok for tok in tokenizer.tokenize(inp_text, wakati=True)]

    word_ids = []
    for word in words:
        idx = input_field.vocab.stoi[word]
        word_ids.append(idx)

    x = torch.tensor(word_ids)
    x = x.view(1, -1)
    y = seq2seq.predict(x, max_length)

    rep_text = ""
    for j in range(y.size()[1]):
        word = reply_field.vocab.itos[y[0][j]]
        if word == "<eos>":
            break
        rep_text += word

    rep_text = rep_text.replace("<sos>", "")
    rep_text = rep_text.replace("<eos>", "")
    rep_text = rep_text.replace("<pad>", "")
    rep_text = rep_text.replace("<unk>", "")

    return rep_text


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
