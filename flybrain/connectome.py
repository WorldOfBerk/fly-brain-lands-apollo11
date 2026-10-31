"""Loads the output of scripts/prepare_connectome.py and maps cell types to neuron indices."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


class Connectome:
    def __init__(self, proc_dir: Path = PROC, weights: str = "calibrated"):
        """weights='calibrated': output of scripts/calibrate_connectome.py (rescaled to FlyWire,
        neuromodulators not treated as fast synapses). weights='raw': W[post, pre] = sign * synapse count."""
        self.neurons = pd.read_parquet(proc_dir / "neurons.parquet")
        name = "weights_calibrated_csr.npz" if weights == "calibrated" else "weights_csr.npz"
        self.W = sp.load_npz(proc_dir / name).tocsr()
        self._types = self.neurons["type"].fillna("")

    @property
    def n(self) -> int:
        return len(self.neurons)

    def idx(self, *types: str, side: str | None = None) -> np.ndarray:
        """Indices of neurons of the given cell types. side: 'L' / 'R' / None."""
        mask = self._types.isin(types)
        if side is not None:
            mask &= self.neurons["somaSide"] == side
        return np.flatnonzero(mask.values)

    def label(self, i: int) -> str:
        row = self.neurons.iloc[i]
        return row["instance"] if isinstance(row["instance"], str) else str(row["bodyId"])
