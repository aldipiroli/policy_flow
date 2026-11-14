import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import zarr

from policy_flow.datasets.base_dataset import BaseDataset


class PushtDataset(BaseDataset):
    def __init__(self, cfg, logger, split="train"):
        super().__init__(cfg, logger, split)
        self.root_dir = Path(self.root_dir)
        self.chunk_size = cfg.chunk_size
        assert self.chunk_size > 0

        zarr_path = self.download()
        val_episodes, train_episodes = self.load_data(zarr_path)
        episodes = train_episodes if self.split == "train" else val_episodes
        self.indices = self.build_indices(episodes)
        self.logger.info(f"PushtDataset [{self.split}]: {len(self.indices)} samples")

    def load_data(self, zarr_path):
        root = zarr.open(str(zarr_path), mode="r")
        self.states = np.asarray(root["data"]["state"][:], dtype=np.float32)
        self.actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
        self.episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        self.episode_starts = np.concatenate(([0], self.episode_ends[:-1]))

        episode_ids = np.random.default_rng(self.cfg.seed).permutation(len(self.episode_ends))
        num_val = min(max(1, round(len(episode_ids) * self.cfg.val_ratio)), len(episode_ids) - 1)
        val_episodes, train_episodes = episode_ids[:num_val], episode_ids[num_val:]
        return val_episodes, train_episodes

    def download(self):
        # https://github.com/berkeleydeeprlcourse/homework_spring2026/blob/main/hw1/src/hw1_imitation/data.py
        PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
        ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"
        zarr_path = self.root_dir / ZARR_RELATIVE_PATH
        if zarr_path.exists():
            return zarr_path
        if not self.cfg.download:
            raise FileNotFoundError(f"Push-T dataset not found at {zarr_path}; set download=true to fetch it")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.root_dir / "pusht.zip"
        if not zip_path.exists():
            self.logger.info("Downloading Push-T dataset")
            partial_path = zip_path.with_suffix(".zip.part")
            urllib.request.urlretrieve(PUSHT_URL, partial_path)
            partial_path.replace(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(self.root_dir)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Push-T archive did not contain {ZARR_RELATIVE_PATH}")
        return zarr_path

    def build_indices(self, episodes):
        indices = []
        for episode in sorted(episodes):
            start = self.episode_starts[episode]
            end = self.episode_ends[episode]
            for t in range(start, end - self.chunk_size + 1):
                indices.append(t)
        return np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        state = self.states[t]  # [agent_x, agent_y, block_x, block_y, block_angle]
        state = torch.from_numpy(state).float()

        action_chunk = self.actions[t : t + self.chunk_size]  # (chunk_size, 2): agent_x, agent_y
        action_chunk = torch.from_numpy(action_chunk).float()
        return state, action_chunk
