#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import zlib
import zipfile
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.spatial.distance import cdist

# ======================= Path Configuration =======================
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = REPO_ROOT / "data"

CACHE_DIR = OUTPUT_BASE / "Raw_Spectral_Cache"
FEATURE_DIR = OUTPUT_BASE / "Extracted_Features"
CORRUPT_LOG = OUTPUT_BASE / "corrupted_spectral_list.txt"

CONTACT_CUTOFF = 10.0
NUM_WORKERS = 8
OVERWRITE = False
# ==================================================================


def save_atomic_npz(target_path: Path, **arrays):
    """Atomic file writing with read-back verification to prevent corrupted outputs."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    np.savez_compressed(temp_path, **arrays)
    try:
        with np.load(temp_path, allow_pickle=False) as chk:
            for k in arrays:
                _ = chk[k].shape
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    os.replace(temp_path, target_path)


def minmax(arr, is_matrix=False):
    """Robust Min-Max normalization."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = (arr > 0) if is_matrix else np.ones(arr.shape, dtype=bool)
    if is_matrix:
        np.fill_diagonal(mask, False)
    res = np.zeros_like(arr, dtype=np.float32)
    if mask.any():
        low, high = arr[mask].min(), arr[mask].max()
        if high > low:
            res[mask] = (arr[mask] - low) / (high - low)
    return res


def get_mode_slices(n_modes):
    """Generate mode index slices across low- and high-frequency bands."""
    slices = {}
    ladder = (10, 20, 30, 40, 60)

    for c in ladder:
        if c <= n_modes:
            slices[f"first{c}"] = slice(0, c)
            slices[f"last{c}"] = slice(max(0, n_modes - c), n_modes)

    top2 = max(1, int(np.ceil(n_modes * 0.02)))
    slices["first2pct"] = slice(0, top2)
    slices["last2pct"] = slice(max(0, n_modes - top2), n_modes)
    slices["all"] = slice(0, n_modes)
    return slices


def derive_rmsf_plugin(stem, coords, anm_vals, anm_u, gnm_vals, gnm_vecs):
    """Derive RMSF features for ANM and GNM across all frequency slices."""
    out_file = FEATURE_DIR / "RMSF" / f"{stem}_rmsf.npz"
    if out_file.is_file() and not OVERWRITE:
        return "skipped"

    anm_slices = get_mode_slices(len(anm_vals))
    gnm_slices = get_mode_slices(len(gnm_vals))
    data = {}

    for name, slc in anm_slices.items():
        u_sub = anm_u[:, :, slc]
        val_sub = np.maximum(anm_vals[slc], 1e-12)
        inv_val = 1.0 / val_sub
        msf = np.einsum('ikm,ikm,m->i', u_sub, u_sub, inv_val)
        rmsf = np.sqrt(np.maximum(msf, 0.0)).reshape(-1, 1).astype(np.float32)
        norm_rmsf = minmax(rmsf)

        data[f"anm_{name}"] = rmsf
        data[f"anm_{name}_minmax"] = norm_rmsf
        data[f"rmsf_anm_{name}"] = rmsf
        data[f"rmsf_anm_{name}_minmax"] = norm_rmsf

    for name, slc in gnm_slices.items():
        vec_sub = gnm_vecs[:, slc]
        val_sub = np.maximum(gnm_vals[slc], 1e-12)
        inv_val = 1.0 / val_sub
        msf = np.einsum('im,im,m->i', vec_sub, vec_sub, inv_val)
        rmsf = np.sqrt(np.maximum(msf, 0.0)).reshape(-1, 1).astype(np.float32)
        norm_rmsf = minmax(rmsf)

        data[f"gnm_{name}"] = rmsf
        data[f"gnm_{name}_minmax"] = norm_rmsf
        data[f"rmsf_gnm_{name}"] = rmsf
        data[f"rmsf_gnm_{name}_minmax"] = norm_rmsf

    save_atomic_npz(out_file, **data)
    return "ok"


def derive_cc_plugin(stem, coords, anm_vals, anm_u, contact_mask):
    """Derive Cross-Correlation (CC) matrices filtered by contact mask."""
    out_file = FEATURE_DIR / "CC" / f"{stem}_cc.npz"
    if out_file.is_file() and not OVERWRITE:
        return "skipped"

    anm_slices = get_mode_slices(len(anm_vals))
    data = {}
    target_groups = (
        "first10", "first20", "first30", "first40", "first60",
        "last10", "last20", "last30", "last40", "last60",
        "first2pct", "last2pct", "all"
    )

    for name in target_groups:
        if name not in anm_slices:
            continue
        u_sub = anm_u[:, :, anm_slices[name]]
        norms = np.sqrt(np.einsum('ikm,ikm->i', u_sub, u_sub))
        norms = np.maximum(norms, 1e-8)
        dots = np.tensordot(u_sub, u_sub, axes=([1, 2], [1, 2]))
        cc = np.clip(dots / np.outer(norms, norms), -1.0, 1.0).astype(np.float32)

        weighted = np.where(contact_mask, np.abs(cc), 0.0).astype(np.float32)
        np.fill_diagonal(weighted, 1.0)
        norm_cc = minmax(weighted, is_matrix=True)
        np.fill_diagonal(norm_cc, 1.0)

        data[f"cc_{name}_raw"] = cc
        data[f"cc_{name}_norm"] = norm_cc
        data[f"cc_{name}"] = norm_cc

    save_atomic_npz(out_file, **data)
    return "ok"


def derive_prs_plugin(stem, coords, anm_vals, anm_u, contact_mask):
    """Derive Perturbation Response Scanning (PRS) effectiveness and sensitivity."""
    out_file = FEATURE_DIR / "PRS" / f"{stem}_prs.npz"
    if out_file.is_file() and not OVERWRITE:
        return "skipped"

    n_res = coords.shape[0]
    anm_slices = get_mode_slices(len(anm_vals))
    data = {}
    target_groups = (
        "first10", "first20", "first30", "first40", "first60",
        "last10", "last20", "last30", "last40", "last60",
        "first2pct", "last2pct", "all"
    )

    for name in target_groups:
        if name not in anm_slices:
            continue
        slc = anm_slices[name]
        u_sub = anm_u[:, :, slc]
        val_sub = np.maximum(anm_vals[slc], 1e-12)

        weighted_u = u_sub / np.sqrt(val_sub)
        M = weighted_u.shape[-1]
        flat_w = weighted_u.transpose(0, 2, 1).reshape(n_res, M * 3)
        G = flat_w @ flat_w.T
        prs = np.sqrt(np.maximum(G, 1e-12)).astype(np.float32)
        del flat_w, G

        prs_no_diag = prs.copy()
        np.fill_diagonal(prs_no_diag, 0.0)
        eff = (np.sum(prs_no_diag, axis=1) / max(1, n_res - 1)).astype(np.float32)
        sens = (np.sum(prs_no_diag, axis=0) / max(1, n_res - 1)).astype(np.float32)
        del prs_no_diag

        eff_norm = minmax(eff.reshape(-1, 1), is_matrix=False)
        sens_norm = minmax(sens.reshape(-1, 1), is_matrix=False)

        prs_contact = prs.copy()
        prs_contact[~contact_mask] = 0.0
        prs_norm = minmax(prs_contact, is_matrix=True)
        del prs_contact

        data[f"prs_{name}"] = prs_norm
        data[f"eff_{name}"] = eff
        data[f"sens_{name}"] = sens
        data[f"prs_eff_{name}"] = eff
        data[f"prs_sens_{name}"] = sens
        data[f"eff_{name}_minmax"] = eff_norm
        data[f"sens_{name}_minmax"] = sens_norm
        data[f"prs_eff_{name}_minmax"] = eff_norm
        data[f"prs_sens_{name}_minmax"] = sens_norm
        del prs

    save_atomic_npz(out_file, **data)
    return "ok"


def derive_stiffness_plugin(stem, coords, anm_vals, anm_u, contact_mask):
    """Analytically compute mechanical deformation stiffness matrix from normal modes."""
    out_file = FEATURE_DIR / "Stiffness" / f"{stem}_stiffness.npz"
    if out_file.is_file() and not OVERWRITE:
        return "skipped"

    N = coords.shape[0]
    c_mask = contact_mask.copy()
    np.fill_diagonal(c_mask, False)
    c_mask = np.triu(c_mask)
    src, dst = np.where(c_mask)

    stiffness = np.zeros((N, N), dtype=np.float32)

    if len(src) > 0:
        w_factor = 1.0 / np.sqrt(np.maximum(anm_vals, 1e-8))
        u_w = anm_u * w_factor[None, None, :]

        CHUNK = 4096
        for s in range(0, len(src), CHUNK):
            e = slice(s, s + CHUNK)
            diff_u = u_w[src[e]] - u_w[dst[e]]
            var_ij = np.einsum('ekm,ekm->e', diff_u, diff_u)
            stiffness[src[e], dst[e]] = (3.0 / np.maximum(var_ij, 1e-6)).astype(np.float32)

        stiffness = stiffness + stiffness.T

    norm_stiff = minmax(stiffness, is_matrix=True)
    np.fill_diagonal(norm_stiff, 1.0)

    save_atomic_npz(out_file, stiffness=stiffness, stiffness_norm=norm_stiff)
    return "ok"


def derive_invariants_plugin(stem, anm_u, anm_vals):
    """Derive SO(3) rotation- and sign-invariant spectral descriptors."""
    out_file = FEATURE_DIR / "Invariants_P0" / f"{stem}_invariants.npz"
    if out_file.is_file() and not OVERWRITE:
        return "skipped"

    n_res, _, total_modes = anm_u.shape
    k_low = min(30, total_modes)
    k_high_start = max(0, total_modes - 30)

    def extract_band_invariants(u_band):
        feats = np.zeros((n_res, 5), dtype=np.float32)
        for i in range(n_res):
            ui = u_band[i, :, :]
            Pi = ui @ ui.T
            evals = np.linalg.eigvalsh(Pi)
            evals = np.maximum(evals, 1e-8)
            tr = np.sum(evals)
            aniso = (evals[2] - evals[0]) / tr
            feats[i] = [evals[0], evals[1], evals[2], tr, aniso]
        return feats

    feats_low = extract_band_invariants(anm_u[:, :, :k_low])
    feats_high = extract_band_invariants(anm_u[:, :, k_high_start:total_modes])
    node_dyn = np.concatenate([feats_low, feats_high], axis=-1).astype(np.float32)

    save_atomic_npz(out_file, node_dyn_10d=node_dyn)
    return "ok"


def process_protein_features(spectral_file_str: str):
    spec_path = Path(spectral_file_str)
    stem = spec_path.stem.replace("_spectral", "")

    try:
        with np.load(spec_path, allow_pickle=False) as data:
            coords = data["coords"]
            anm_vals = data["anm_eigvals"]
            anm_u = data["anm_u"]
            gnm_vals = data["gnm_eigvals"]
            gnm_vecs = data["gnm_vecs"]
    except (zipfile.BadZipFile, EOFError, zlib.error, ValueError, KeyError, OSError) as corrupt_err:
        return "corrupted", f"{stem} corrupted: {corrupt_err}"
    except Exception as e:
        return "failed", f"{stem} read failed: {e}"

    try:
        contact_mask = (cdist(coords, coords, metric="sqeuclidean") <= CONTACT_CUTOFF ** 2)

        s1 = derive_rmsf_plugin(stem, coords, anm_vals, anm_u, gnm_vals, gnm_vecs)
        s2 = derive_cc_plugin(stem, coords, anm_vals, anm_u, contact_mask)
        s3 = derive_prs_plugin(stem, coords, anm_vals, anm_u, contact_mask)
        s4 = derive_stiffness_plugin(stem, coords, anm_vals, anm_u, contact_mask)
        s5 = derive_invariants_plugin(stem, anm_u, anm_vals)

        if all(x == "skipped" for x in (s1, s2, s3, s4, s5)):
            return "skipped", f"{stem} already exists"
        return "ok", f"{stem} completed"
    except MemoryError:
        return "oom", f"{stem} out of memory"
    except Exception as e:
        return "failed", f"{stem} failed: {e}"


def main():
    t0 = time.time()
    print(f"=== [Tier 2] Starting Feature Plugin Derivation ===")
    print(f"Spectral Cache Dir: {CACHE_DIR}")
    print(f"Feature Output Dir: {FEATURE_DIR}")

    for sub in ("RMSF", "CC", "PRS", "Stiffness", "Invariants_P0"):
        (FEATURE_DIR / sub).mkdir(parents=True, exist_ok=True)

    cache_files = sorted(CACHE_DIR.glob("*_spectral.npz"))
    if not cache_files:
        raise FileNotFoundError(f"No spectral cache files found in {CACHE_DIR}!")

    print(f"Found {len(cache_files)} cached proteins. Running with {NUM_WORKERS} workers...")
    counts = Counter()
    corrupt_stems = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(process_protein_features, str(f)): f for f in cache_files}
        for i, future in enumerate(as_completed(future_to_file), 1):
            f_path = future_to_file[future]
            try:
                status, msg = future.result()
            except Exception as pool_err:
                status, msg = "worker_died", f"{f_path.stem} worker terminated: {pool_err}"

            counts[status] += 1
            if status in ("corrupted", "worker_died"):
                corrupt_stems.append(f_path.stem)

            if i % 1000 == 0 or status in ("failed", "corrupted", "worker_died", "oom"):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(cache_files) - i)
                print(f"[{i}/{len(cache_files)}] {f_path.stem}: {status} -> {msg} "
                      f"(Elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min)", flush=True)

    if corrupt_stems:
        CORRUPT_LOG.write_text("\n".join(corrupt_stems), encoding="utf-8")
        print(f"\nWarning: {len(corrupt_stems)} corrupted files logged to {CORRUPT_LOG}")

    print(f"\n=== [Tier 2] Derivation Summary (Total Time {(time.time()-t0)/60:.1f} min) ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
