import torch
import scipy.io as scio
from tqdm import tqdm

from configs.config import get_config
from Dataset import Dataset
from Runner import Runner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_config = get_config(opts=[
    'TRAIN.STEP_SIZE', 200
])


for rat in ['028']:
    for region in ['M1']:  # ['mPFC', 'M1']:
        pbar = tqdm(desc=f'{rat} {region}', total=35)
        for test_fold in range(0, 5):
            # get data: use train set as test set
            dataset = Dataset(default_config, rat, region, test_fold, device, False)
            dataset.test_in = dataset.train_in
            dataset.test_out = dataset.train_out

            #for latent_dim in [1, 2, 3, 4, 8, 16, 24]:
            for latent_dim in [24]:
                embed_dim = latent_dim if latent_dim > 16 else 16
                if rat == '025':  # the number of  025's data samples allows more parameters
                    embed_dim = 24
                file_prefix = f'results/tVAE_{rat}_{region}_{test_fold}_1En1De_{latent_dim}latent'

                # get model
                config = get_config('explore_latent_with_decoder_pos', [
                    'DATA.RAT', rat,
                    'DATA.REGION', region,
                    'DATA.TEST_FOLD', test_fold,
                    'MODEL.LATENT_DIM', latent_dim,
                    'MODEL.EMBED_DIM', embed_dim,
                    'TRAIN.BATCH_SIZE_TEST', 64 * 16,
                ])
                runner = Runner(config, dataset, config.MODEL.VARIATIONAL)
                checkpoint = torch.load(f'{file_prefix}_decoder_pos.pth', map_location=device) # load checkpoint
                runner.model.load_state_dict(checkpoint['model_state_dict'])

                # get results
                results = runner.evaluate(dataset.train_movements, dataset.train_trial_no)

                # save results
                scio.savemat(f'{file_prefix}_decoder_pos_train_set.mat', results)

                # update tqdm
                pbar.update(1)

        pbar.close()
