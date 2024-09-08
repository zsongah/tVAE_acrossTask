import os
import scipy.io as scio
from scipy.ndimage import gaussian_filter1d
import numpy as np
# import time
from matplotlib import pyplot as plt
import matplotlib.style

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.functional

from tqdm import tqdm

from data_preparation_realData_suc import action_colors, get_batch_random, get_batch_ss
from model.models import VAE, initialize_weights
import Dataset

matplotlib.use('agg')
matplotlib.style.use('fast')


class Runner:

    def __init__(self, config, dataset: Dataset, variational=True):
        super(Runner, self).__init__() # 允许子类调用父类的方法

        self.config = config
        self.data = dataset
        self.device = dataset.device
        self.beta = 0. # 
        self.variational = variational

        self.model = VAE(config, self.device, self.data.in_neuron_num,
                         self.data.out_neuron_num, self.variational).to(self.device) # 将模型参数都移动到指定设备上。

        self.optimizer = AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())), #返回模型中可学习的参数
            lr=config.TRAIN.LR.INIT,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        self.scheduler = SequentialLR(self.optimizer, [
                LinearLR(
                    self.optimizer,
                    start_factor=1e-6,
                    end_factor=1,
                    total_iters=config.TRAIN.LR.WARMUP
                ),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.TRAIN.NUM_UPDATES - config.TRAIN.LR.WARMUP,
                ),
            ], milestones=[10])

    def run(self, test_movements, test_trial_no, resume_file, target_file):
        if not os.path.exists(self.config.CHECKPOINT_DIR):
            os.makedirs(self.config.CHECKPOINT_DIR)
        if not os.path.exists(self.config.RESULT_DIR):
            os.makedirs(self.config.RESULT_DIR)
        if not os.path.exists(self.config.FIG_DIR):
            os.makedirs(self.config.FIG_DIR)

        self.model.apply(initialize_weights) #将 initialize_weights 函数应用于 self.model 的所有子模块。
        train_loss_all = []
        val_loss_all = []
        start_epoch = 0
        save_dict = None

        # best_val_loss = float("inf")
        # best_model = None

        # resume training
        if os.path.isfile(self.config.CHECKPOINT_DIR + resume_file + '.pth'):
            checkpoint = torch.load(self.config.CHECKPOINT_DIR + resume_file + '.pth', map_location=self.device)
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            train_loss_all = checkpoint['train_loss_all']
            val_loss_all = checkpoint['val_loss_all']
            save_dict = {
                'epoch': start_epoch,
                'config': self.config,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss_all': train_loss_all,
                'val_loss_all': val_loss_all,
            }
            print(f'Loaded checkpoint from epoch {start_epoch}')
        # tqdm 用于提供进度条
        for epoch in tqdm(range(start_epoch + 1, self.config.TRAIN.NUM_UPDATES + 1), desc='Training', unit='epoch'):
            # epoch_start_time = time.time()
            train_loss = self.train_1_epoch(epoch)
            train_loss_all.extend(train_loss)
            self.scheduler.step() # 更新学习率调度器额状态

            if self.variational and epoch > 30:
                self.beta = self.config.TRAIN.BETA * ((epoch - 10) % 20)

            if epoch % self.config.TRAIN.VAL_INTERVAL == 0:
                results = self.evaluate(test_movements, test_trial_no)

                if epoch % self.config.TRAIN.VAL_DRAW_INTERVAL == 0:
                    self.plot_result(results, self.config.CHECKPOINT_DIR, target_file, epoch,
                                     has_var=self.variational, need_show=self.config.TRAIN.SHOW_PLOTS)

                val_loss_all.append(results["loss"])
            #     elapsed = time.time() - epoch_start_time
            #     print('-' * 89)
            #     print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            #           f'valid loss {results["loss"][0]:5.5f}')
            # print('-' * 89)

            # if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_model = model

            save_dict = {
                'epoch': epoch,
                'config': self.config,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss_all': train_loss_all,
                'val_loss_all': val_loss_all,
            }

            if epoch % self.config.TRAIN.CHECKPOINT_INTERVAL == 0:
                torch.save(save_dict, f'{self.config.CHECKPOINT_DIR}{target_file}_checkpoint_{epoch}.pth')

        print(f'| End of training | train loss {train_loss_all[-1][0]:5.5f} | valid loss {val_loss_all[-1][0]:5.5f}')
        torch.save(save_dict, f'{self.config.RESULT_DIR}{target_file}.pth')
        results = self.evaluate(test_movements, test_trial_no)
        self.plot_result(results, self.config.FIG_DIR, target_file, self.config.TRAIN.NUM_UPDATES,
                         has_var=self.variational, need_show=self.config.TRAIN.SHOW_PLOTS)
        results["train_loss_all"] = train_loss_all
        results["val_loss_all"] = val_loss_all
        scio.savemat(f'{self.config.RESULT_DIR}/{target_file}.mat', results)

        fig, axs = plt.subplots(3)
        loss_type = ["total loss", "BCE", "KLD"]
        for i in range(0, len(train_loss_all[-1])):
            axs[i].plot(np.arange(1, len(train_loss_all)+1)/self.config.TRAIN.LOGS_PER_EPOCH,
                        np.array(train_loss_all)[:, i], label='train loss')
            axs[i].plot(np.arange(self.config.TRAIN.VAL_INTERVAL, self.config.TRAIN.NUM_UPDATES+1,
                                  self.config.TRAIN.VAL_INTERVAL),
                        np.array(val_loss_all)[:, i], label='test loss')
            axs[i].set_ylabel(loss_type[i])
        axs[2].set_xlabel('epoch')
        axs[2].legend(loc="upper right")

        plt.savefig(f'{self.config.FIG_DIR}/{target_file}_learning_curve.png', bbox_inches='tight')

        plt.cla()
        plt.close('all')

    def train_1_epoch(self, epoch) -> list:
        batch_size = self.config.TRAIN.BATCH_SIZE
        train_in = self.data.train_in
        train_out = self.data.train_out

        self.model.train()  # turn on train mode
        total_loss = [0., 0., 0.]
        loss_log = []
        # start_time = time.time()

        segment_num = train_in.size(1)
        num_batches = segment_num // batch_size
        log_interval = num_batches // self.config.TRAIN.LOGS_PER_EPOCH
        indices = torch.randperm(segment_num).tolist()
        for batch, start_idx in enumerate(range(0, segment_num, batch_size)):
            data, _ = get_batch_random(train_in, batch_size, indices, start_idx)
            _, targets = get_batch_random(train_out, batch_size, indices, start_idx)

            output, mu, log_var = self.model(data)
            output_flat = output.permute(1, 0, 2).reshape(-1, self.data.out_neuron_num)
            loss, bce, kld = self.model.loss_function(output_flat, targets, mu, log_var, self.beta)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAIN.CLIP_GRAD_NORM)
            self.optimizer.step()

            for i, item in enumerate([loss, bce, kld]):
                total_loss[i] += item.item()

            if batch % log_interval == 0 and batch > 0:
                # lr = self.optimizer.param_groups[0]['lr']
                # ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                if len(loss_log) == 0:
                    # cur_loss = total_loss[0] / (log_interval + 1)
                    loss_log.append([total_loss[i] / (log_interval + 1) for i in range(3)])
                else:
                    # cur_loss = total_loss[0] / log_interval
                    loss_log.append([total_loss[i] / log_interval for i in range(3)])

                # print(f'| epoch {epoch:3d} | {batch:2d}/{num_batches:2d} batches | '
                #       f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                #       f'beta {self.beta:.3f} | loss {cur_loss:5.5f}')
                total_loss = [0., 0., 0.]
                # start_time = time.time()

        return loss_log

    def evaluate(self, eval_events: Tensor, eval_trial_no: Tensor) -> (list, dict):
        latent_dim = self.config.MODEL.LATENT_DIM
        batch_size = self.config.TRAIN.BATCH_SIZE_TEST
        step_size = self.config.TRAIN.STEP_SIZE_TEST

        self.model.eval()  # turn on evaluation mode
        total_loss = [0., 0., 0.]
        predictions = np.empty((0, self.data.out_neuron_num))
        truth = np.empty((0, self.data.out_neuron_num))
        movements = np.empty((0, 1), dtype=int)
        trials = np.empty((0, 1), dtype=int)
        latent_mu = np.empty((0, latent_dim))
        latent_std = np.empty((0, latent_dim))
        with torch.no_grad():
            for start_idx in range(0, self.data.test_in.size(1), batch_size):
                data, _, _ = get_batch_ss(self.data.test_in, batch_size, start_idx, step_size)
                _, _, targets = get_batch_ss(self.data.test_out, batch_size, start_idx, step_size)
                _, _, events = get_batch_ss(eval_events, batch_size, start_idx, step_size)
                _, _, trial_no = get_batch_ss(eval_trial_no, batch_size, start_idx, step_size)

                output, mu, log_var = self.model(data)
                output_valid = output[-step_size:, :].permute(1, 0, 2).reshape(-1, self.data.out_neuron_num)
                mu_valid = mu[-step_size:, :]
                log_var_valid = log_var[-step_size:, :]

                loss, bce, kld = self.model.loss_function(output_valid, targets, mu_valid, log_var_valid, self.beta)

                for i, item in enumerate([loss, bce, kld]):
                    total_loss[i] += output_valid.size(0) * item.item()
                predictions = np.vstack((predictions, output_valid.cpu().numpy()))
                truth = np.vstack((truth, targets.cpu().numpy()))
                movements = np.vstack((movements, events.numpy()))
                trials = np.vstack((trials, trial_no.numpy()))
                latent_mu = np.vstack(
                    (latent_mu, mu_valid.permute(1, 0, 2).reshape(-1, latent_dim).cpu().numpy()))
                latent_std = np.vstack(
                    (latent_std, torch.exp(0.5 * log_var_valid.permute(1, 0, 2).reshape(-1, latent_dim)).cpu().numpy()))

        predicted_spikes = np.random.binomial(1, predictions)
        loss = [total_loss[i] / len(predictions) for i in range(3)]
        results = {'loss': loss,
                   'truth': truth,
                   'predictions': predictions,
                   'predicted_spikes': predicted_spikes,
                   'movements': movements,
                   'trials': trials,
                   'latent_mu': latent_mu,
                   'latent_std': latent_std,
                   }

        return results

    @staticmethod # 静态方法,不需要实例化就可以调用
    def plot_result(results, save_dir, target_file, epoch, has_var, need_show):
        plot_time = np.arange(0, 30, 0.01)
        plot_indexes = range(0, 3000)
        fig, axs = plt.subplots(4, 2)
        for i in range(3):
            for j in range(2):
                # plot actions
                colors = action_colors[results["movements"][plot_indexes, 0].astype(int)]
                axs[i, j].bar(plot_time, np.ones(len(plot_indexes)), color=colors, bottom=0, alpha=0.5)
                # plot firing rate of neuron i
                rate = gaussian_filter1d(results["truth"][plot_indexes, i * 2 + j], sigma=10)
                axs[i, j].plot(plot_time, rate, label='Ground Truth', color='k')
                axs[i, j].plot(plot_time, results["predictions"][plot_indexes, i * 2 + j],
                               label='Predictions', color='g')
                # plot spike trains of neuron i
                spike_times = plot_time[results["truth"][plot_indexes, i * 2 + j] == 1]
                axs[i, j].vlines(spike_times, 0.75, 0.95, color='k', linewidth=0.3, label='Truth Spikes')
                spike_times = plot_time[results["predicted_spikes"][plot_indexes, i * 2 + j] == 1]
                axs[i, j].vlines(spike_times, 0.5, 0.7, color='g', linewidth=0.3, label='Predicted Spikes')
                axs[i, j].set_title('Neuron ' + str(i * 2 + j))
                axs[i, j].set_xticklabels([])
        axs[2, 0].set_ylabel('Firing Probability')
        axs[0, 1].legend(loc='upper right', bbox_to_anchor=(1.8, 1.2))

        for d in range(results["latent_mu"].shape[1]):
            axs[3, 0].plot(plot_time, results["latent_mu"][plot_indexes, d])
            if has_var:
                axs[3, 0].fill_between(
                    plot_time,
                    (results["latent_mu"][plot_indexes, d] - results["latent_std"][plot_indexes, d]).squeeze(),
                    (results["latent_mu"][plot_indexes, d] + results["latent_std"][plot_indexes, d]).squeeze(),
                    alpha=.1)

            axs[3, 1].plot(plot_time, gaussian_filter1d(results["latent_mu"][plot_indexes, d], sigma=10))
            if has_var:
                axs[3, 1].fill_between(
                    plot_time,
                    (gaussian_filter1d(results["latent_mu"][plot_indexes, d], sigma=10) - 1).squeeze(),
                    (gaussian_filter1d(results["latent_mu"][plot_indexes, d], sigma=10) + 1).squeeze(),
                    alpha=.1)
            axs[3, 0].set_xlabel('Time (sec)')

        plt.savefig(f'{save_dir}/{target_file}_{epoch}.png', bbox_inches='tight')

        if need_show:
            plt.show()

