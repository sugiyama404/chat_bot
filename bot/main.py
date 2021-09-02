from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dill
from janome.tokenizer import Tokenizer

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST', 'GET'])
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
        self.dropout = dropout  # ドロップアウト層

        # 埋め込み層
        self.embedding = nn.Embedding(n_vocab, n_emb)
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.gru = nn.GRU(  # GRU層
            input_size=n_emb,  # 入力サイズ
            hidden_size=n_h,  # ニューロン数
            batch_first=True,  # 入力を (バッチサイズ, 時系列の数, 入力の数) にする
            num_layers=num_layers,  # RNN層の数（層を重ねることも可能）
            bidirectional=bidirectional,  # Bidrectional RNN
        )

    def forward(self, x):
        # 文章の長さを取得
        idx_pad = input_field.vocab.stoi["<pad>"]
        sentence_lengths = x.size()[1] - (x == idx_pad).sum(dim=1)

        y = self.embedding(x)  # 単語をベクトルに変換
        y = self.embedding_dropout(y)
        y = nn.utils.rnn.pack_padded_sequence(  # 入力のパッキング
            y,
            sentence_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        y, h = self.gru(y)

        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)  # テンソルに戻す
        if self.bidirectional:  # 双方向の値を足し合わせる
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

        # 埋め込み層
        self.embedding = nn.Embedding(n_vocab, n_emb)
        self.embedding_dropout = nn.Dropout(self.dropout)  # ドロップアウト層

        self.gru = nn.GRU(  # GRU層
            input_size=n_emb,  # 入力サイズ
            hidden_size=n_h,  # ニューロン数
            batch_first=True,  # 入力を (バッチサイズ, 時系列の数, 入力の数) にする
            num_layers=num_layers,  # RNN層の数（層を重ねることも可能）
        )

        self.fc = nn.Linear(n_h*2, self.n_out)  # コンテキストベクトルが合流するので2倍のサイズ

    def forward(self, x, h_encoder, y_encoder):
        y = self.embedding(x)  # 単語をベクトルに変換
        y = self.embedding_dropout(y)
        y, h = self.gru(y, h_encoder)

        # ----- Attension -----
        y_tr = torch.transpose(y, 1, 2)  # 次元1と次元2を入れ替える
        ed_mat = torch.bmm(y_encoder, y_tr)  # バッチごとに行列積
        attn_weight = F.softmax(ed_mat, dim=1)  # attension weightの計算
        attn_weight_tr = torch.transpose(attn_weight, 1, 2)  # 次元1と次元2を入れ替える
        context = torch.bmm(attn_weight_tr, y_encoder)  # コンテキストベクトルの計算
        y = torch.cat([y, context], dim=2)  # 出力とコンテキストベクトルの合流

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

    def forward(self, x_encoder, x_decoder):  # 訓練に使用
        if self.is_gpu:
            x_encoder, x_decoder = x_encoder.cuda(), x_decoder.cuda()

        batch_size = x_decoder.shape[0]
        n_time = x_decoder.shape[1]
        y_encoder, h = self.encoder(x_encoder)

        y_decoder = torch.zeros(batch_size, n_time, self.decoder.n_out)
        if self.is_gpu:
            y_decoder = y_decoder.cuda()

        for t in range(0, n_time):
            x = x_decoder[:, t:t+1]  # Decoderの入力を使用
            y, h = self.decoder(x, h, y_encoder)
            y_decoder[:, t:t+1, :] = y
        return y_decoder

    def predict(self, x_encoder, max_length=10):  # 予測に使用
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
            x = y  # 前の時刻の出力を入力に
            if self.is_gpu:
                x = x.cuda()
            y, h = self.decoder(x, h, y_encoder)
            y = y.argmax(2)
            y_decoder[:, t:t+1] = y
        return y_decoder


# ------ モデルの読み込み ------
input_field = torch.load("input_field.pkl", pickle_module=dill)
reply_field = torch.load("reply_field.pkl", pickle_module=dill)

is_gpu = False  # GPUを使用するかどうか
n_h = 896
n_vocab_inp = len(input_field.vocab.itos)
n_vocab_rep = len(reply_field.vocab.itos)
n_emb = 300
n_out = n_vocab_rep
early_stop_patience = 5  # 早期終了のタイミング（誤差の最小値が何回更新されなかったら終了か）
num_layers = 1
bidirectional = True
dropout = 0.0
clip = 100

encoder = Encoder(n_h, n_vocab_inp, n_emb, num_layers,
                  bidirectional, dropout=dropout)
decoder = Decoder(n_h, n_out, n_vocab_rep, n_emb, num_layers, dropout=dropout)
seq2seq = Seq2Seq(encoder, decoder, is_gpu=is_gpu)

seq2seq.load_state_dict(torch.load(
    "model_bot.pth", map_location=torch.device("cpu")))  # CPU対応

# ------ 応答文の生成 ------
j_tk = Tokenizer()


def reply(inp_text, tokenizer, max_length=10):
    words = [tok for tok in tokenizer.tokenize(inp_text, wakati=True)]  # 分かち書き

    # 入力を単語idの並びに変換
    word_ids = []
    for word in words:
        idx = input_field.vocab.stoi[word]
        word_ids.append(idx)

    x = torch.tensor(word_ids)
    x = x.view(1, -1)  # バッチ対応
    y = seq2seq.predict(x, max_length)

    # 応答文の作成
    rep_text = ""
    for j in range(y.size()[1]):
        word = reply_field.vocab.itos[y[0][j]]
        if word == "<eos>":
            break
        rep_text += word

    # トークンの削除
    rep_text = rep_text.replace("<sos>", "")
    rep_text = rep_text.replace("<eos>", "")
    rep_text = rep_text.replace("<pad>", "")
    rep_text = rep_text.replace("<unk>", "")

    return rep_text


if __name__ == "__main__":
    # db.create_all()
    app.run(host='0.0.0.0', port=8000)
