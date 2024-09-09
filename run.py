# import os
# import scipy.io as scio
import random
import numpy as np
import torch
# from torchinfo import summary

from Dataset import Dataset
from configs.config import get_config
from Runner import Runner


def run_exp(config, dataset, paths):
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    experiment = f'tVAE_{config.DATA.RAT}_{config.DATA.REGION}_{config.DATA.TEST_FOLD}_' \
                 f'{config.MODEL.N_LAYERS_ENCODER}En' \
                 f'{config.MODEL.N_LAYERS_DECODER}De_' \
                 f'{config.MODEL.LATENT_DIM}latent_' \
                 f'{paths}'
    print(f'Experiment {experiment} start.')
    runner = Runner(config, dataset, config.MODEL.VARIATIONAL)
    runner.run(test_movements=dataset.test_movements, test_trial_no=dataset.test_trial_no,
               resume_file=f'{experiment}', target_file=experiment)
    print('\n')

    # print('+' * 89)
    # print(f'Training on {config.DATA.REGION} fold {config.DATA.TEST_FOLD} Start')
    # print('+' * 89 + '\n')

    # summary(runner.model,
    #         (config.MODEL.TIME_WINDOW, config.TRAIN.BATCH_SIZE, runner.data.in_neuron_num),
    #         device=runner.device)

    # print('\n\n\n')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_config = get_config()
    # '''
    # Debug the code
    # '''
    # region = 'mPFC'
    # test_fold = 0
    # dataset = Dataset(default_config, rat, region, test_fold, device)
    # latent_dim = 16
    # embed_dim = 16
    # config = get_config('debug', [
    #     'DATA.RAT', rat,
    #     'DATA.REGION', region,
    #     'DATA.TEST_FOLD', test_fold,
    #     'MODEL.LATENT_DIM', latent_dim,
    #     'MODEL.EMBED_DIM', embed_dim
    # ])
    # run_exp(config, dataset, 'debug')

    '''
    explore latent with positional encoding in tVAE decoder
    '''
    # config the parameters
    rat = '028'
    for region in ['M1']:  # ['mPFC', 'M1']:
        for test_fold in range(0, 5):
            dataset = Dataset(default_config, rat, region, test_fold, device)
            for latent_dim in [24, 16, 8, 4, 3, 2, 1]:  # [6]:
                embed_dim = latent_dim if latent_dim > 16 else 16
                if rat == '025':  # the number of  025's data samples allows more parameters
                    embed_dim = 24
                config = get_config('explore_latent_with_decoder_pos', [
                    'DATA.RAT', rat,
                    'DATA.REGION', region,
                    'DATA.TEST_FOLD', test_fold,
                    'MODEL.LATENT_DIM', latent_dim,
                    'MODEL.EMBED_DIM', embed_dim,
                ])
                run_exp(config, dataset, 'decoder_pos')


if __name__ == "__main__":
    main()
    # test
    train()



