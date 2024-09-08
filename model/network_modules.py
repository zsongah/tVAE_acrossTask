from typing import Tuple, Union
import math
import torch
from torch import nn, Tensor
import torch.nn.functional


class Encoder(nn.Module):

    def __init__(self, config, device, neuron_num: int, need_var: bool = False):
        super(Encoder, self).__init__()
        self.model_type = 'Dynamic_Encoder'
        self.device = device
        self.need_var = need_var
        self.d_model = config.MODEL.EMBED_DIM
        self.neuron_num = neuron_num

        d_model = config.MODEL.EMBED_DIM
        dropout = config.MODEL.DROPOUT
        n_head = config.MODEL.NUM_HEADS
        d_hid = config.MODEL.HIDDEN_SIZE
        n_layers_encoder = config.MODEL.N_LAYERS_ENCODER
        latent_dim = config.MODEL.LATENT_DIM

        # Preprocess
        self.embedding = nn.Linear(neuron_num, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers_encoder)

        # Latent space
        self.fc_mu = nn.Linear(d_model, latent_dim)
        if need_var:
            self.fc_var = nn.Linear(d_model, latent_dim)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoded_src = self.transformer_encoder(src, src_mask)
        mu = self.fc_mu(encoded_src)

        if self.need_var:
            log_var = self.fc_var(encoded_src)
        else:
            log_var = torch.zeros_like(mu)

        return mu, log_var


class Decoder(nn.Module):

    def __init__(self, config, device, neuron_num: int, pos=False):
        super(Decoder, self).__init__()
        self.model_type = 'Dynamic_Decoder'
        self.device = device
        self.d_model = config.MODEL.EMBED_DIM
        self.neuron_num = neuron_num
        self.pos = pos

        latent_dim = config.MODEL.LATENT_DIM # dz
        d_model = config.MODEL.EMBED_DIM # d
        n_head = config.MODEL.NUM_HEADS
        d_hid = config.MODEL.HIDDEN_SIZE
        dropout = config.MODEL.DROPOUT
        n_layers_decoder = config.MODEL.N_LAYERS_DECODER

        self.fc_z = nn.Linear(latent_dim, d_model) 
        if self.pos:
            self.pos_encoder = PositionalEncoding(d_model, dropout) # embeding + positional encoding
        decoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, n_layers_decoder)
        self.linear = nn.Linear(d_model, neuron_num) # d->N
        self.outputLayer = nn.Sigmoid()

    def forward(self, z: Tensor, z_mask: Tensor = None) -> Tensor:
        if z_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            z_mask = nn.Transformer.generate_square_subsequent_mask(len(z)).to(self.device)
        z = self.fc_z(z)
        if self.pos:
            z = self.pos_encoder(z)
        decoded_z = self.transformer_decoder(z, z_mask)
        return self.outputLayer(self.linear(decoded_z))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # so that pe will not be considered as weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
