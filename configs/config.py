from typing import List, Optional, Union
from yacs.config import CfgNode

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

_C = CfgNode()
_C.SEED = 42
_C.CHECKPOINT_DIR = "checkpoints/"
_C.RESULT_DIR = "results/"
_C.FIG_DIR = "figs/"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.RAT = ''
_C.DATA.REGION = ''
_C.DATA.TEST_FOLD = 0
_C.DATA.SELECT_M1 = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.NAME = "NeuralDataTransformerVAE"
_C.MODEL.VARIATIONAL = True
_C.MODEL.HIDDEN_SIZE = 32  # dimension of the feedforward network model in ``nn.TransformerEncoder``
_C.MODEL.DROPOUT = .1  # dropout probability TODO: different dropout rate for different parts?
_C.MODEL.NUM_HEADS = 2  # number of heads for multi-headed self-attention``
_C.MODEL.N_LAYERS_ENCODER = 1  # number of ``nn.TransformerEncoderLayer`` for VAE encoder
_C.MODEL.N_LAYERS_DECODER = 1  # number of ``nn.TransformerEncoderLayer`` for VAE decoder
_C.MODEL.DECODER_POS = False  # whether decoder use positional encoding
_C.MODEL.EMBED_DIM = 16  # embedding dimension. Note that here is total but NDT is per neuron
_C.MODEL.LATENT_DIM = 3  # latent dimension
_C.MODEL.TIME_WINDOW = 300  # time window

# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.BATCH_SIZE_TEST = 64 * 8
_C.TRAIN.STEP_SIZE = 10
_C.TRAIN.STEP_SIZE_TEST = 200
_C.TRAIN.NUM_UPDATES = 100  # Max updates
_C.TRAIN.CLIP_GRAD_NORM = 0.5

_C.TRAIN.LR = CfgNode()
_C.TRAIN.LR.INIT = 5e-3
_C.TRAIN.LR.SCHEDULE = True
_C.TRAIN.LR.SCHEDULER = "cosine"
_C.TRAIN.LR.WARMUP = 10
_C.TRAIN.WEIGHT_DECAY = 0.01

_C.TRAIN.BETA = 0.005  # note that the actual maximum beta is BETA * 19

_C.TRAIN.CHECKPOINT_INTERVAL = 4
_C.TRAIN.LOGS_PER_EPOCH = 4
_C.TRAIN.VAL_INTERVAL = 4
_C.TRAIN.VAL_DRAW_INTERVAL = 24  # must be times of VAL_INTERVAL
_C.TRAIN.SHOW_PLOTS = False


def get_cfg_defaults():
    """Get default config (yacs config node)."""
    return _C.clone()


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CfgNode:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list, e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(DEFAULT_CONFIG_DIR + config_path + '.yaml')

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
