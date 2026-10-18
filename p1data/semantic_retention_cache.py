"""Lossless disk-backed crops with the original Semantic Drone crop schedule."""
import json
from pathlib import Path
import numpy as np
import torch

from p1data.torch_dataset import random_aligned_crop


class SemanticRetentionCache:
    def __init__(self, root, seed, *, allow_unverified=False):
        self.root = Path(root); self.seed = seed; self.epoch = 0
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        allowed = ('verified_lossless_cache', 'cache_equivalence_pending') if allow_unverified else ('verified_lossless_cache',)
        if self.manifest['status'] not in allowed or len(self.manifest['pairs']) != 400:
            raise ValueError('require a complete verified 400-pair cache')
        self.pairs = self.manifest['pairs']

    def __len__(self):
        return len(self.pairs)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('negative crop nonce')
        self.epoch = epoch

    def __getitem__(self, index):
        record = self.pairs[index]
        image = np.load(self.root/record['image'], mmap_mode='r', allow_pickle=False)
        target = np.load(self.root/record['target'], mmap_mode='r', allow_pickle=False)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        image, target = random_aligned_crop(image, target, 512, rng)
        return {'image': torch.from_numpy(np.array(image, dtype=np.uint8, copy=True, order='C')).permute(2, 0, 1).float().div_(255.),
                'target': torch.from_numpy(np.array(target, dtype=np.int64, copy=True, order='C')),
                'identifier': record['identifier']}
