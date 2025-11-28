import torch
import torch.nn as nn
import math

class LuongAttention(nn.Module):
    def _init_(self, hidden_dim):
        super()._init_()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        # score = decoder_hidden @ W @ encoder_outputs.T
        dec = self.linear(decoder_hidden).unsqueeze(2)  # (b,h,1)
        scores = torch.bmm(encoder_outputs, dec).squeeze(2)  # (b,seq)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (b,h)
        return context, weights

class AttnLSTM(nn.Module):
    def _init_(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1, out_steps=10, out_dim=None):
        super()._init_()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attn = LuongAttention(hidden_dim)
        self.out_steps = out_steps
        self.out_dim = out_dim if out_dim is not None else input_dim
        # decoder: simple MLP that uses context + last decoder hidden to predict next step autoregressively
        self.dec_input = nn.Linear(self.out_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x, teacher_forcing_y=None):
        # x: (b, seq, input_dim)
        b = x.size(0)
        enc_out, (h_n, c_n) = self.lstm(x)  # enc_out: (b, seq, h)
        # use last hidden as decoder init
        dec_hidden = h_n[-1].unsqueeze(0)  # (1, b, h) for GRU
        # start token: last observed step
        last_obs = x[:, -1, :]  # (b, feat)
        preds = []
        dec_input = self.dec_input(last_obs)  # (b,h)
        dec_input = dec_input.unsqueeze(1)  # (b,1,h)
        for t in range(self.out_steps):
            out, dec_hidden = self.decoder_rnn(dec_input, dec_hidden)
            # out: (b,1,h) -> squeeze
            out_s = out.squeeze(1)
            # attention over encoder outputs using out_s
            context, attn_w = self.attn(out_s, enc_out)
            combined = out_s + context
            step_pred = self.proj(combined)  # (b, out_dim)
            preds.append(step_pred.unsqueeze(1))
            # next decoder input: use step_pred
            dec_input = self.dec_input(step_pred).unsqueeze(1)
        preds = torch.cat(preds, dim=1)  # (b, out_steps, out_dim)
        return preds

class TransformerForecast(nn.Module):
    def _init_(self, input_dim, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, out_steps=10, out_dim=None):
        super()._init_()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_steps = out_steps
        self.out_dim = out_dim if out_dim is not None else input_dim
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, self.out_dim * out_steps)
        )

    def forward(self, x):
        # x (b, seq, input_dim)
        proj = self.input_projection(x)  # (b, seq, d_model)
        enc = self.transformer(proj)  # (b, seq, d_model)
        # pool: use last token representation (or mean)
        pooled = enc.mean(dim=1)
        out = self.decoder(pooled)
        out = out.view(x.size(0), self.out_steps, self.out_dim)
        return out
