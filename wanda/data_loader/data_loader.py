from sklearn.utils import shuffle
from wanda import config
from torch.utils.data import DataLoader, ConcatDataset
from wanda.data_loader.dataset import HSWifiTrainDataset


def create_hs_data_loader(batch_size, train=True, shuffle=True, greyscale=False):
    ds = HSWifiTrainDataset(train=train, greyscale=greyscale)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=config.N_WORKER, shuffle=shuffle
    )


def create_combined_dataloader(batch_size, shuffle=True, greyscale=False):
    train = HSWifiTrainDataset(train=True, greyscale=greyscale)
    test = HSWifiTrainDataset(train=False, greyscale=greyscale)
    dataset_combined = ConcatDataset([train, test])
    return DataLoader(
        dataset_combined,
        batch_size=batch_size,
        num_workers=config.N_WORKER,
        shuffle=shuffle,
    )

