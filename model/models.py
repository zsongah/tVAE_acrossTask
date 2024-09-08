from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional
from model.network_modules import Encoder, Decoder


class VAE(nn.Module): # nn.module is the base class for all neural network modules in pytorch
    # VAE include a Encoder and a Decoder
    def __init__(self, config, device, in_neuron_num: int, out_neuron_num: int, variational: bool = True):
        super(VAE, self).__init__()
        self.model_type = 'AutoEncoder'
        self.config = config
        self.device = device
        self.in_neuron_num = in_neuron_num
        self.out_neuron_num = out_neuron_num
        self.variational = variational

        latent_dim = config.MODEL.LATENT_DIM

        self.encoder = Encoder(config, device, in_neuron_num, need_var=variational) #定义了模型结构和forward函数
        self.decoder = Decoder(config, device, out_neuron_num, self.config.MODEL.DECODER_POS)#定义了模型结构和forward函数

        # latent smooth
        sigma = 10
        kernel_size = 2 * 4 * sigma + 1
        self.padding_size = 4 * sigma
        kernel = get_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).repeat(latent_dim, 1, 1)
        self.register_buffer('kernel', kernel) # so that pe will not be considered as weights

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        if src_mask is None:  # TODO: use a mask to control the number of points of look back? use triu maybe
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)

        mu, log_var = self.encoder(src, src_mask)
        if self.variational and self.training:  # do not sample in testing
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return self.decoder(z, src_mask), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta):
        bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')

        var_priori = torch.ones_like(log_var) * 1
        mu_priori = nn.functional.conv1d(
            nn.functional.pad(mu.detach().permute(1, 2, 0), (self.padding_size, self.padding_size), mode='reflect'),
            self.kernel,
            padding='valid', groups=mu.size(2)).permute(2, 0, 1)

        mu_priori *= 0.999

        kld = -0.5 * torch.mean(
            1 + log_var - var_priori.log() - log_var.exp() / var_priori - (mu - mu_priori).pow(2) / var_priori)
        return bce + beta * kld, bce, kld


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def get_kernel(size: int, sigma: float):
    # x = torch.arange(-size // 2 + 1., size // 2 + 1.).cuda()
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = x / sigma
    kernel = torch.exp(-0.5 * x ** 2)
    kernel = kernel / kernel.sum()
    return kernel


# class AE(nn.Module):
#
#     def __init__(self, config, device, in_neuron_num: int, out_neuron_num: int):
#         super(AE, self).__init__()
#         self.model_type = 'AutoEncoder'
#         self.config = config
#         self.device = device
#         self.in_neuron_num = in_neuron_num
#         self.out_neuron_num = out_neuron_num
#
#         self.encoder = Encoder(config, device, in_neuron_num, need_var=False)
#         self.decoder = Decoder(config, device, out_neuron_num)
#
#     def forward(self, src: Tensor, src_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
#         """
#         Arguments:
#             src: Tensor, shape ``[seq_len, batch_size, in_neuron_num]``
#             src_mask: Tensor, shape ``[seq_len, seq_len]``
#
#         Returns:
#             tuple (output, mu), where output has shape ``[seq_len, batch_size, out_neuron_num]`` and mu
#             has shape ``[seq_len, batch_size, latent_dim]``
#         """
#
#         if src_mask is None:
#             """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
#             Unmasked positions are filled with float(0.0).
#             """
#             src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
#
#         mu = self.encoder(src, src_mask)
#         output = self.decoder(mu, src_mask)
#
#         return output, mu
#
#     @staticmethod
#     def loss_function(recon_x, x):
#         return nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
#
#
