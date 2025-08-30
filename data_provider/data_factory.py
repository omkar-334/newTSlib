from functools import partial

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_M4,
    MSLSegLoader,
    PSMSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWaTSegLoader,
    UEAloader,
)
from data_provider.uea import collate_fn

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "m4": Dataset_M4,
    "PSM": PSMSegLoader,
    "MSL": MSLSegLoader,
    "SMAP": SMAPSegLoader,
    "SMD": SMDSegLoader,
    "SWaT": SWaTSegLoader,
    "UEA": UEAloader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = True if args.shuffle_test else flag not in {"test", "TEST"}
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    # Initialize dataset
    if args.task_name == "anomaly_detection":
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
    elif args.task_name == "classification":
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )
        collate = partial(collate_fn, max_len=args.seq_len)
    else:
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate if args.task_name == "classification" else None,
        # pin_memory=True,
        persistent_workers=True,  # 🔹 keep workers alive across epochs
        prefetch_factor=4,  # 🔹 prefetch batches for faster loading
    )

    return data_set, data_loader
