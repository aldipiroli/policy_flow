from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, cfg, logger, split="train"):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.split = "train" if split == "train" else "val"
        self.root_dir = cfg.root_dir

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
