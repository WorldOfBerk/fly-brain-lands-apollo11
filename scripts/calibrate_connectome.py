"""Rescales MaleCNS weights to the calibration of the Shiu model (FlyWire 783).

Why: Shiu et al. tuned w_syn = 0.275 mV/synapse to FlyWire synapse counts. MaleCNS counts more synapses
per neuron for the same cell types, and with raw weights the model fell into a persistent state of
~20k active neurons (all Kenyon cells on) after any stimulus.

Fixes, each based on data or published biology rather than free tuning:
  1) Per-type scaling: each postsynaptic neuron's inputs are divided by the MaleCNS/FlyWire ratio of
     mean input synapses for its cell type (via MaleCNS flywireType). Unmatched types (e.g. VNC) use
     the median ratio.
  2) Neuromodulators (dopamine, serotonin, octopamine) are not fast synapses (weight 0). They act through
     slow metabotropic receptors; the Shiu model treated them as excitatory and they supplied ~29 % of
     the excitatory current in the persistent state.
  3) On the model side (flybrain.LIFParams.conductance): conductance-based synapses, E_exc = 0 mV,
     E_inh = -75 mV.

Output: data/processed/weights_calibrated_csr.npz, data/processed/calibration_report.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MOD_NT = ["dopamine", "serotonin", "octopamine"]


def main():
    neurons = pd.read_parquet(PROC / "neurons.parquet")
    W = sp.load_npz(PROC / "weights_csr.npz").tocsr()

    ann = pd.read_feather(ROOT / "data/malecns_v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    fw_type = ann.set_index("bodyId")["flywireType"].reindex(neurons["bodyId"]).values
    in_m = np.asarray(np.abs(W).sum(axis=1)).ravel()

    con = pd.read_parquet(ROOT / "data/flywire_783/Connectivity_783.parquet")
    comp = pd.read_csv(ROOT / "data/flywire_783/Completeness_783.csv", index_col=0)
    fw = pd.read_csv(ROOT / "data/flywire_783/neuron_annotations.tsv", sep="\t", low_memory=False).set_index("root_id")
    in_f = np.bincount(con["Postsynaptic_Index"].values, weights=con["Connectivity"].values, minlength=len(comp))
    f_mean = pd.DataFrame({"t": fw["cell_type"].reindex(comp.index).values, "in": in_f}).dropna().groupby("t")["in"].mean()
    m_mean = pd.DataFrame({"t": fw_type, "in": in_m}).dropna().groupby("t")["in"].mean()
    both = pd.concat([m_mean.rename("m"), f_mean.rename("f")], axis=1, join="inner")
    both = both[both.f > 20]  # ratio is noisy for types with few inputs
    ratio = both.m / both.f
    median = float(ratio.median())
    per_neuron = pd.Series(fw_type).map(ratio).fillna(median).values

    W = W.tocoo()
    modulatory = neurons["nt"].isin(MOD_NT).values
    keep = ~modulatory[W.col]
    data = W.data[keep] / per_neuron[W.row[keep]]
    Wc = sp.csr_matrix((data.astype(np.float32), (W.row[keep], W.col[keep])), shape=W.shape)
    sp.save_npz(PROC / "weights_calibrated_csr.npz", Wc)

    report = {
        "matched_types": int(len(ratio)),
        "ratio_median": median,
        "ratio_quartiles": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))],
        "neurons_scaled_by_type_ratio": int(pd.Series(fw_type).map(ratio).notna().sum()),
        "neurons_scaled_by_median": int(pd.Series(fw_type).map(ratio).isna().sum()),
        "removed_neuromodulator_synapses": int(np.abs(W.data[~keep]).sum()),
        "removed_neuromodulator_neurons": int(modulatory.sum()),
    }
    (PROC / "calibration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
