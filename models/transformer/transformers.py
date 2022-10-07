import torch.nn as nn
import torch
import torch.nn.functional as F
from models.transformer.attention import MultiheadAttention2

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward(self, src, src_mask, tgt, tgt_mask):
        non_pad_src_mask = None if src_mask is None else 1 - src_mask
        non_pad_tgt_mask = None if tgt_mask is None else 1 - tgt_mask

        if src is not None:
            src = src.transpose(0, 1)

        x = tgt.transpose(0, 1)
        for layer in self.decoder_layers:
            x, weight = layer(x, non_pad_tgt_mask,
                              src, non_pad_src_mask,
                              self.buffered_future_mask(x))
        return x.transpose(0, 1)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        self.self_attn = MultiheadAttention2(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MultiheadAttention2(d_model, num_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, encoder_out=None, encoder_mask=None, self_attn_mask=None):
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        if encoder_out is not None:
            res = x
            x, weight = self.encoder_attn(x, encoder_out, encoder_out, encoder_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = res + x
            x = self.encoder_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x, weight


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        non_padding_mask = None if mask is None else 1 - mask
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, non_padding_mask)
        x = x.transpose(0, 1)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        self.attn_mask = None

        self.self_attn = MultiheadAttention2(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        dim = x.size(0)

        attn_mask = None if self.attn_mask is None else self.attn_mask.cuda()[:dim, :dim]
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dropout)

    def forward(self, src, src_mask, tgt, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(enc_out, src_mask, tgt, tgt_mask)
        return out


class DualTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2, dropout=0.0):
        super().__init__()
        self.decoder1 = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)
        self.decoder2 = TransformerDecoder(num_decoder_layers2, d_model, num_heads, dropout)

    def forward(self, src1, src_mask1, src2, src_mask2, decoding, enc_out=None):
        assert decoding in [1, 2]
        if decoding == 1:
            if enc_out is None:
                enc_out = self.decoder2(None, None, src2, src_mask2)
            out = self.decoder1(enc_out, src_mask2, src1, src_mask1)
        elif decoding == 2:
            if enc_out is None:
                enc_out = self.decoder1(None, None, src1, src_mask1)
            out = self.decoder2(enc_out, src_mask1, src2, src_mask2)
        return enc_out, out