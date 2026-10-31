"""Builds a signed weight matrix from the raw MaleCNS v1.0 release.

Output:
  data/processed/neurons.parquet  - neuron table in index order (bodyId, type, superclass, nt, sign)
  data/processed/weights_csr.npz  - W[post, pre] = sign(pre) * synapse count (scipy CSR)
  data/processed/soma_um.npy      - soma positions in um (NaN where unknown), for the brain view

Sign rule as in Shiu et al. (2024): GABA and glutamate inhibitory, everything else excitatory.
Histamine (photoreceptors) is also treated as inhibitory.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path(__file__).resolve().parents[1] / "data" / "malecns_v1.0"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
INHIBITORY = {"gaba", "glutamate", "histamine"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    ann = pd.read_feather(RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ann = ann[ann["status"] == "Traced"]
    neurons = ann[["bodyId", "type", "instance", "superclass", "class", "somaSide"]].reset_index(drop=True)

    nt = pd.read_feather(RAW / "body-neurotransmitters-male-cns-v1.0.feather")
    nt = nt.set_index("body")
    consensus = nt["consensus_nt"].where(nt["consensus_nt"] != "unclear", nt["predicted_nt"])
    neurons["nt"] = neurons["bodyId"].map(consensus).fillna("unclear")
    neurons["sign"] = np.where(neurons["nt"].isin(INHIBITORY), -1, 1).astype(np.int8)

    index = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"])
    w = pd.read_feather(RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    w = w[w["body_pre"].isin(index.index) & w["body_post"].isin(index.index)]
    pre = index[w["body_pre"].values].values
    post = index[w["body_post"].values].values
    vals = w["weight"].values.astype(np.float32) * neurons["sign"].values[pre]

    n = len(neurons)
    W = sp.csr_matrix((vals, (post, pre)), shape=(n, n))
    W.sum_duplicates()

    neurons.to_parquet(OUT / "neurons.parquet")
    sp.save_npz(OUT / "weights_csr.npz", W)

    soma = ann.set_index("bodyId")["somaLocation"].reindex(neurons["bodyId"]).values
    soma = np.array([np.nan * np.ones(3) if v is None or np.ndim(v) == 0 else np.asarray(v, float) for v in soma])
    np.save(OUT / "soma_um.npy", (soma * 8 / 1000).astype(np.float32))  # 8 nm voxels

    print(f"neurons: {n:,}  connections: {W.nnz:,}  synapses: {int(np.abs(W.data).sum()):,}")
    print(neurons["nt"].value_counts().to_string())


if __name__ == "__main__":
    main()
