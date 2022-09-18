from wanda import config
from torch.utils.data import DataLoader
from wanda.data_loader.dataset import HSWifiTrainDataset


def create_hs_data_loader(batch_size):
    ds = HSWifiTrainDataset()
    return DataLoader(ds, batch_size=batch_size, num_workers=config.N_WORKER)
