import torch
import numpy as np

from configs.config import get_config
from model.models import VAE


default_config = get_config()
seq_len = default_config.MODEL.TIME_WINDOW
step_size = default_config.TRAIN.STEP_SIZE_TEST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)


def get_model(rat, test_fold, latent_dim, region, neuron_num):
    test_fold = int(test_fold)
    latent_dim = int(latent_dim)
    neuron_num = int(neuron_num)

    embed_dim = latent_dim if latent_dim > 16 else 16
    if rat == '025':  # the number of  025's data samples allows more parameters
        embed_dim = 24

    config = get_config(opts=[
        'DATA.RAT', rat,
        'DATA.REGION', region,
        'DATA.TEST_FOLD', test_fold,
        'MODEL.LATENT_DIM', latent_dim,
        'MODEL.EMBED_DIM', embed_dim,
        'MODEL.DECODER_POS', True,
        'TRAIN.BATCH_SIZE_TEST', 64 * 16,
    ])

    model = VAE(config, device, neuron_num, neuron_num, True).to(device)
    checkpoint = torch.load(f'results/tVAE_{rat}_{region}_{test_fold}_1En1De_{latent_dim}latent_decoder_pos.pth',
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def latent2neural(model, data, neuron_num):
    neuron_num = int(neuron_num)
    data = torch.tensor(np.asarray(data), dtype=torch.float32)

    latent = segment(data).to(device)
    with torch.no_grad():
        output = model.decoder(latent, src_mask)
        output_valid = output[-step_size:, :].permute(1, 0, 2).reshape(-1, neuron_num)
        predictions = np.vstack((output[0:-step_size, 0, :].cpu().numpy(), output_valid.cpu().numpy()))
        predictions = predictions[0:data.size(1), :]

    return predictions


def neural2latent(model, data, latent_dim):
    latent_dim = int(latent_dim)
    data = torch.tensor(np.asarray(data), dtype=torch.float32)

    spikes = segment(data).to(device)
    with torch.no_grad():
        mu, log_var = model.encoder(spikes, src_mask)
        latent_mu = np.vstack((
            mu[0:-step_size, 0, :].cpu().numpy(),
            mu[-step_size:, :].permute(1, 0, 2).reshape(-1, latent_dim).cpu().numpy()
        ))
        latent_std = np.vstack((
            torch.exp(0.5 * log_var[0:-step_size, 0, :]).cpu().numpy(),
            torch.exp(0.5 * log_var[-step_size:, :].permute(1, 0, 2).reshape(-1, latent_dim)).cpu().numpy()
        ))

    return latent_mu, latent_std


def segment(input_data):
    """Process the data into segments with overlapping

    Arguments:
        input_data: Tensor, shape ``[neuron_num, N]``

    Returns:
        Tensor, shape ``[seq_len, segment_num, neuron_num]``
    """
    neuron_num, total_len = input_data.size()
    segment_num = (total_len - seq_len) // step_size + 1
    if (total_len - seq_len) % step_size != 0:
        segment_num += 1  # Add an extra segment if there is remaining data
    segments = np.empty((seq_len, segment_num, neuron_num))

    for seq, i in enumerate(range(0, total_len - seq_len, step_size)):
        segments[:, seq, :] = input_data[:, i: i + seq_len].t()

    return torch.FloatTensor(segments)
