# -*- coding: utf-8 -*-
"""SimMatrix: npy-backed drop-in replacement for chars_sim_dict.

Original code does json `chars_sim_dict[trg_uni][ref_uni] -> float`. For 5,855
chars the full JSON would be ~34M entries, so we store an [N,N] fp32 matrix and
expose the same nested-dict interface. Unknown chars fall back to 0.0 (neutral
weight under softmax) instead of KeyError.
"""
import json

import numpy as np


class _Row:
    def __init__(self, row, idx):
        self._row, self._idx = row, idx

    def __getitem__(self, uni):
        i = self._idx.get(uni)
        return float(self._row[i]) if i is not None else 0.0


class SimMatrix:
    def __init__(self, npy_path, unis_json):
        self.sim = np.load(npy_path)
        with open(unis_json, encoding="utf-8") as f:
            unis = json.load(f)
        self.idx = {u: i for i, u in enumerate(unis)}
        assert self.sim.shape[0] == len(self.idx), \
            f"sim {self.sim.shape} vs {len(self.idx)} unis"

    def __getitem__(self, uni):
        i = self.idx.get(uni)
        return _Row(self.sim[i] if i is not None else np.zeros(self.sim.shape[1]),
                    self.idx)
