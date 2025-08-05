import sys

# Add the current directory to path to import data_provider modules
sys.path.append(".")

from data_provider.data_factory import data_provider


class Args:
    """Simple args class to replace get_args() for Jupyter usage"""

    def __init__(self):
        # Basic arguments
        self.task_name = "classification"
        self.data = "UEA"
        self.root_path = "./other_datasets/EthanolConcentration/"
        self.model_id = "EthanolConcentration"
        self.batch_size = 16
        self.seq_len = 256
        self.num_workers = 0

        # Additional required arguments
        self.is_training = 1
        self.model = "Autoformer"  # Default model
        self.data_path = ""
        self.features = "M"
        self.target = ""
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = ""
        self.inverse = False
        self.mask_rate = 0.25
        self.anomaly_ratio = 0.25
        self.expand = 0
        self.d_conv = 1
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.05
        self.embed = "timeF"
        self.activation = "gelu"
        self.channel_independence = 0
        self.decomp_method = "moving_avg"
        self.use_norm = 1
        self.down_sampling_layers = 1
        self.down_sampling_window = 1
        self.down_sampling_method = "avg"
        self.seg_len = 6
        self.itr = 1
        self.train_epochs = 100
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = "test"
        self.loss = "MSE"
        self.lradj = "type1"
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.gpu_type = "0"
        self.use_multi_gpu = False
        self.devices = "0,1,2,3"
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2
        self.use_dtw = 0
        self.augmentation_ratio = 0
        self.seed = 42
        self.jitter = False
        self.scaling = False
        self.permutation = False
        self.randompermutation = False
        self.magwarp = False
        self.timewarp = False
        self.windowslice = False
        self.windowwarp = False
        self.rotation = False
        self.spawner = False
        self.dtwwarp = False
        self.shapedtwwarp = False
        self.wdba = False
        self.discdtw = False
        self.discsdtw = False
        self.extra_tag = ""
        self.patch_len = 16
        self.hidden_dim = 128
        self.n_emb = 4
        self.attn_dropout = 0.1
        self.mlp_ratio = 1
        self.n_depth = 2
        self.use_cond = 1
        self.use_tphi = 2
        self.beta_schedule = "cosine"
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.timesteps = 100
        self.shuffle_test = False
        self.sweep = False
        self.wandb = False
        self.normalize = True
        self.classifier = 1
        self.tphi_loss = True
        self.filename = "test"


def load_ethanol_concentration_dataset(batch_size=16, seq_len=256):
    """
    Load the EthanolConcentration dataset using the existing data loading infrastructure.

    Args:
        batch_size (int): Batch size for dataloaders
        seq_len (int): Sequence length for padding

    Returns:
        dict: Dictionary containing train and test dataloaders and dataset info
    """
    # Create args object with the required parameters for EthanolConcentration
    args = Args()

    # Set the required arguments for EthanolConcentration dataset
    args.batch_size = batch_size
    args.seq_len = seq_len

    # Load training data
    print("Loading training data...")
    train_dataset, train_loader = data_provider(args, flag="TRAIN")

    # Load test data
    print("Loading test data...")
    test_dataset, test_loader = data_provider(args, flag="TEST")

    print("Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_names)}")
    print(f"Class names: {train_dataset.class_names}")
    print(f"Max sequence length: {train_dataset.max_seq_len}")
    print(f"Number of features: {train_dataset.feature_df.shape[1]}")

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "class_names": train_dataset.class_names,
        "max_seq_len": train_dataset.max_seq_len,
        "n_features": train_dataset.feature_df.shape[1],
        "n_classes": len(train_dataset.class_names),
    }


if __name__ == "__main__":
    # Load the dataset
    dataset_info = load_ethanol_concentration_dataset(batch_size=16, seq_len=256)
