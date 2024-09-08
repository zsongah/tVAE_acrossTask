import scipy.io as scio
# from scipy.ndimage import gaussian_filter1d
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
# from matplotlib import pyplot as plt
import warnings

action_colors = np.array([[1, 1, 1],
                          [0, 0.4470, 0.7410],
                          [0.9290, 0.6940, 0.1250],
                          [0.6350, 0.0780, 0.1840]])


def get_data(file_path, select_m1):
    # load data
    data = scio.loadmat(file_path)
    mpfc = np.array(data['mPFC'])
    m1 = np.array(data['M1'])
    if select_m1:
        m1 = np.array(data['M1_select'])
    trial_no = np.array(data['trial_No']).squeeze()
    movements = np.array(data['movements'])
    folds = [np.array(fold.squeeze()).astype(np.int16) for fold in data["folds"].squeeze()]

    mpfc = torch.from_numpy(mpfc).float()
    m1 = torch.from_numpy(m1).float()
    trial_no = torch.from_numpy(trial_no)
    movements = torch.from_numpy(movements).float()
    folds = [torch.from_numpy(fold) for fold in folds]

    return_dict = {
        'mPFC': mpfc,
        'M1': m1,
        'movements': movements,
        'trial_No': trial_no,
        'folds': folds
    }

    return return_dict


def prepare_train_test(data, test_fold, log=True):
    test_trials = data["folds"][test_fold]
    train_trials = torch.cat([data["folds"][fold] for fold in range(5) if fold != test_fold])

    train_indices = torch.cat([torch.nonzero(torch.eq(data["trial_No"], i)) for i in train_trials]).squeeze()
    test_indices = torch.cat([torch.nonzero(torch.eq(data["trial_No"], i)) for i in test_trials]).squeeze()

    mpfc_train = data["mPFC"][:, train_indices]
    mpfc_test = data["mPFC"][:, test_indices]
    m1_train = data["M1"][:, train_indices]
    m1_test = data["M1"][:, test_indices]

    movements_train = data["movements"][:, train_indices]
    movements_test = data["movements"][:, test_indices]

    trial_no_train = data["trial_No"][train_indices].unsqueeze(0)
    trial_no_test = data["trial_No"][test_indices].unsqueeze(0)

    if log:
        print(f'| load neural data | '
              f'train length: {m1_train.size(1)}, trial: {len(train_trials)} | '
              f'test length: {m1_test.size(1)} , trial: {len(test_trials)} | '
              f'{mpfc_train.size(0)} mPFC neurons | '
              f'{m1_train.size(0)} M1 neurons | ')

    return_dict = {
        'mPFC_train': mpfc_train,
        'M1_train': m1_train,
        'movements_train': movements_train,
        'trial_No_train': trial_no_train,
        'mPFC_test': mpfc_test,
        'M1_test': m1_test,
        'movements_test': movements_test,
        'trial_No_test': trial_no_test,
        'train_trials': train_trials,
        'test_trials': test_trials
    }

    return return_dict


def segment(input_data, seq_len, step_size):
    """Process the data into segments with overlapping

    Arguments:
        input_data: Tensor, shape ``[neuron_num, N]``
        seq_len: int, time window size
        step_size: interval between the beginning of two sequences

    Returns:
        Tensor, shape ``[seq_len, segment_num, neuron_num]``
    """
    neuron_num, total_len = input_data.size()
    segment_num = (total_len - seq_len - 1) // step_size + 1
    segments = np.empty((seq_len + 1, segment_num, neuron_num))

    for seq, i in enumerate(range(0, total_len - seq_len - 1, step_size)):
        segments[:, seq, :] = input_data[:, i: i + seq_len + 1].t()

    return torch.FloatTensor(segments)


def segment_all(data, time_window, train_step_size, test_step_size):
    data['mPFC_train'] = segment(data['mPFC_train'], time_window, train_step_size)
    data['M1_train'] = segment(data['M1_train'], time_window, train_step_size)
    data['movements_train'] = segment(data['movements_train'], time_window, train_step_size)
    data['trial_No_train'] = segment(data['trial_No_train'], time_window, train_step_size)

    data['mPFC_test'] = segment(data['mPFC_test'], time_window, test_step_size)
    data['M1_test'] = segment(data['M1_test'], time_window, test_step_size)
    data['movements_test'] = segment(data['movements_test'], time_window, test_step_size)
    data['trial_No_test'] = segment(data['trial_No_test'], time_window, test_step_size)

    return data


def get_batch(segments: Tensor, bsz: int, i: int) -> Tuple[Tensor, Tensor]:
    warnings.warn("Use get_batch_random for train and get_batch_ss for test.", DeprecationWarning)
    """
    Args:
        segments: Tensor, shape ``[seq_len, segment_num, neuron_num]``
        bsz: int, batch size
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size, neuron_num]``
        and target has shape ``[seq_len * batch_size, neuron_num]``
    """
    neuron_num = segments.size(2)
    data = segments[0:-1, i:i + bsz]
    target = segments[1:, i:i + bsz].permute(1, 0, 2).reshape(-1, neuron_num)
    return data, target


def get_batch_ss(segments: Tensor, bsz: int, i: int, step_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    ss is short for step size
    Args:
        segments: Tensor, shape ``[seq_len, segment_num, neuron_num]``
        bsz: int, batch size
        i: int, start position of batch
        step_size: int, determine the length of target

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size, neuron_num]``
        and target has shape ``[seq_len * batch_size, neuron_num]``
    """
    segment_num = segments.size(1)
    neuron_num = segments.size(2)
    data = segments[0:-1, i:min(i + bsz, segment_num)]
    target = segments[1:, i:min(i + bsz, segment_num)].permute(1, 0, 2).reshape(-1, neuron_num)
    target_valid = segments[-step_size:, i:min(i + bsz, segment_num)].permute(1, 0, 2).reshape(-1, neuron_num)
    return data, target, target_valid


def get_batch_random(segments: Tensor, bsz: int, indices: list, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        segments: Tensor, shape ``[seq_len, segment_num, neuron_num]``
        bsz: int, batch size
        indices: list, a list of shuffled indices
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size, neuron_num]``
        and target has shape ``[seq_len * batch_size, neuron_num]``
    """
    segment_num = segments.size(1)
    neuron_num = segments.size(2)

    indices = indices[i:min(i + bsz, segment_num)]

    data = segments[0:-1, indices]
    target = segments[1:, indices].permute(1, 0, 2).reshape(-1, neuron_num)
    return data, target
