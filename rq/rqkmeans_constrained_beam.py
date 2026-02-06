#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ-KMeans with Constrained Balanced Clustering
===============================================
Uses k-means-constrained to ensure balanced cluster sizes
"""

import os
import numpy as np
import polars as pl
import time
import argparse
import json
from collections import defaultdict
from tqdm import tqdm

try:
    from k_means_constrained import KMeansConstrained
    HAS_CONSTRAINED = True
except ImportError:
    HAS_CONSTRAINED = False
    # print("Warning: k-means-constrained not available")
    
try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def balanced_kmeans_level_constrained(X, K, max_iter=100, tol=1e-7, random_state=None, verbose=False):
    """Balanced K-means implemented with k-means-constrained or fallback to MiniBatchKMeans"""
    start_time = time.time()
    n, d = X.shape
    X = X.astype(np.float32, copy=False)

    # Threshold for switching to fast approximation
    # k-means-constrained is O(N^2) or worse, so it's very slow for N > 20k
    LARGE_DATA_THRESHOLD = 20000

    use_fast = False
    if n > LARGE_DATA_THRESHOLD:
        if HAS_SKLEARN:
            use_fast = True
            if verbose:
                print(f"    [Notice] Data size {n} > {LARGE_DATA_THRESHOLD}. Switching to MiniBatchKMeans for speed.")
                print("    (Strict balanced constraint is relaxed to approximate balance)")
        else:
            print(f"    [Warning] Data size {n} is large but sklearn is not installed. Training will be very slow.")

    if use_fast:
        # Use MiniBatchKMeans for large datasets
        # Increase batch size for better quality on large data
        batch_size = min(n, 4096 * 4)
        kmeans = MiniBatchKMeans(
            n_clusters=K,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            batch_size=batch_size,
            n_init=3,
            verbose=0
        )
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        if verbose:
            print(f"    Using MiniBatchKMeans with batch_size={batch_size}")

    else:
        # Use k-means-constrained for smaller datasets
        if not HAS_CONSTRAINED:
            raise ImportError("k-means-constrained not installed and dataset is small enough to use it.")
            
        # Calculate min and max cluster size
        min_size = max(1, n // K - 1)  # allow some imbalance
        max_size = n // K + 1
    
        if verbose:
            print(f"    Starting constrained K-means with K={K}, n={n}, d={d}")
            print(f"    Cluster size constraints: [{min_size}, {max_size}]")
    
        # Use k-means-constrained
        kmeans = KMeansConstrained(
            n_clusters=K,
            size_min=min_size,
            size_max=max_size,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=3,
            verbose=verbose,
            n_jobs=16
        )
    
        # Train and get labels
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

    print(f"[Time] balanced_kmeans_level_constrained (K={K}): {time.time() - start_time:.2f}s")

    if verbose:
        # Check cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        # If using fast mode, show imbalance ratio
        if use_fast:
             imbalance = counts.max() / counts.mean()
             print(f"    Imbalance ratio: {imbalance:.2f} (1.0 is perfectly balanced)")

    return labels, centroids


def residual_kmeans_constrained(X, K, L, max_iter=300, tol=1e-4, random_state=None, verbose=False):
    """
    Residual K-means with constrained balanced clustering

    Args:
        X: Input data (N, d)
        K: Number of clusters per level (int or list)
        L: Number of levels
        max_iter: Maximum iterations for K-means
        tol: Convergence tolerance
        random_state: Random seed
        verbose: Print detailed info

    Returns:
        codes_all: (L, N) integer codes for each level
        codebooks: List of L codebooks, each (K, d)
        recon: Reconstructed data (N, d)
    """
    total_start = time.time()
    n, d = X.shape
    Ks = ([K] * L) if isinstance(K, int) else list(K)
    assert len(Ks) == L

    X = X.astype(np.float32, copy=False)
    R = X.copy()
    codes_all = np.empty((L, n), dtype=np.int32)
    codebooks = []

    for l in tqdm(range(L), desc="Residual K-means Levels"):
        level_start = time.time()
        k_l = Ks[l]
        if verbose:
            mse_before = np.mean(R ** 2)
            print(f"\n=== Level {l+1}/{L} | K={k_l} ===")
            print(f"  Residual MSE before clustering: {mse_before:.6f}")

        # Generate random seed for sub-level
        seed_l = None if random_state is None else int(np.random.RandomState(random_state + l).randint(0, 2**31 - 1))

        codes_l, C_l = balanced_kmeans_level_constrained(
            R, k_l, max_iter=max_iter, tol=tol, random_state=seed_l, verbose=verbose
        )

        codes_all[l] = codes_l
        codebooks.append(C_l)

        # Subtract reconstructed part from residual
        R -= C_l[codes_l]

        print(f"[Time] Level {l+1}: {time.time() - level_start:.2f}s")
        if verbose:
            mse_after = np.mean(R ** 2)
            print(f"  Residual MSE after Level {l+1}: {mse_after:.6f}")

    recon = X - R
    print(f"[Time] residual_kmeans_constrained total: {time.time() - total_start:.2f}s")

    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"\nFinal reconstruction MSE: {total_mse:.6f}")

    return codes_all, codebooks, recon


def encode_topk_beam_search(X, codebooks, topk=4, beam_width=16, verbose=False):
    """
    Top-K × 路径组合编码 (Multi-path RQ with Beam Search)
    
    核心思想：在每层不做立即的硬选择，而是把 Top-K 作为"分叉点"，
    让路径树在早期展开，最终选择全局重构误差最小的路径。
    
    Args:
        X: Input data (N, d)
        codebooks: List of L codebooks, each (K, d)
        topk: Number of top candidates to consider at each level
        beam_width: Maximum number of paths to keep (beam search width)
        verbose: Print detailed info
    
    Returns:
        codes_all: (L, N) integer codes for each level (best path for each sample)
        recon: Reconstructed data (N, d)
    """
    start_time = time.time()
    n, d = X.shape
    L = len(codebooks)
    X = X.astype(np.float32, copy=False)
    
    if verbose:
        print(f"\n=== Top-K Beam Search Encoding ===")
        print(f"  Samples: {n}, Levels: {L}, TopK: {topk}, BeamWidth: {beam_width}")
    
    # Final codes for all samples
    codes_all = np.empty((L, n), dtype=np.int32)
    recon = np.zeros_like(X)
    
    # Process each sample
    for i in tqdm(range(n), desc="Beam Search Encoding", disable=not verbose):
        x = X[i]  # (d,)
        
        # Each path is represented as: (path_codes, cumulative_reconstruction, total_error)
        # path_codes: list of code indices for each level
        # cumulative_reconstruction: sum of selected centroids
        # total_error: squared reconstruction error
        
        # Initialize with empty path
        # Format: [(codes_list, reconstruction_vector, error), ...]
        active_paths = [([], np.zeros(d, dtype=np.float32), 0.0)]
        
        for l in range(L):
            C_l = codebooks[l]  # (K, d)
            K_l = C_l.shape[0]
            new_paths = []
            
            for path_codes, recon_so_far, _ in active_paths:
                # Current residual
                residual = x - recon_so_far  # (d,)
                
                # Compute distances to all centroids in this level
                # dist[j] = ||residual - C_l[j]||^2
                dists = np.sum((residual[None, :] - C_l) ** 2, axis=1)  # (K,)
                
                # Get top-k nearest centroids
                k_actual = min(topk, K_l)
                topk_indices = np.argpartition(dists, k_actual - 1)[:k_actual]
                
                for j in topk_indices:
                    new_codes = path_codes + [j]
                    new_recon = recon_so_far + C_l[j]
                    # Total reconstruction error after this level
                    new_error = np.sum((x - new_recon) ** 2)
                    new_paths.append((new_codes, new_recon, new_error))
            
            # Beam pruning: keep only top beam_width paths by error
            if len(new_paths) > beam_width:
                # Sort by error and keep best
                new_paths.sort(key=lambda p: p[2])
                active_paths = new_paths[:beam_width]
            else:
                active_paths = new_paths
        
        # Select the best path (minimum reconstruction error)
        best_path = min(active_paths, key=lambda p: p[2])
        best_codes, best_recon, _ = best_path
        
        # Store results
        for l in range(L):
            codes_all[l, i] = best_codes[l]
        recon[i] = best_recon
    
    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"  Beam Search MSE: {total_mse:.6f}")
    
    print(f"[Time] encode_topk_beam_search: {time.time() - start_time:.2f}s")
    return codes_all, recon


def encode_topk_beam_search_batch(X, codebooks, topk=4, beam_width=16, batch_size=1000, verbose=False):
    """
    Batched version of Top-K Beam Search encoding for better performance.
    
    Uses vectorized operations within batches to speed up computation.
    
    Args:
        X: Input data (N, d)
        codebooks: List of L codebooks, each (K, d)
        topk: Number of top candidates to consider at each level
        beam_width: Maximum number of paths to keep
        batch_size: Number of samples to process together
        verbose: Print detailed info
    
    Returns:
        codes_all: (L, N) integer codes
        recon: Reconstructed data (N, d)
    """
    start_time = time.time()
    n, d = X.shape
    L = len(codebooks)
    X = X.astype(np.float32, copy=False)
    
    if verbose:
        print(f"\n=== Batched Top-K Beam Search Encoding ===")
        print(f"  Samples: {n}, Levels: {L}, TopK: {topk}, BeamWidth: {beam_width}")
    
    codes_all = np.empty((L, n), dtype=np.int32)
    recon = np.zeros_like(X)
    
    # Process in batches
    num_batches = (n + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Beam Search Encoding (Batched)"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n)
        X_batch = X[start_idx:end_idx]  # (B, d)
        B = X_batch.shape[0]
        
        # For each sample in batch, maintain beam_width paths
        # Shape: (B, beam_width) for each attribute
        # We'll use lists of arrays for flexibility
        
        # Initialize: one empty path per sample
        # paths_codes[b] = list of (beam_width,) arrays for each level
        # paths_recon[b] = (beam_width, d) reconstruction vectors
        # paths_error[b] = (beam_width,) errors
        
        # Simplified: process each sample in batch individually but with optimized inner loop
        for local_i in range(B):
            global_i = start_idx + local_i
            x = X_batch[local_i]
            
            active_paths = [([], np.zeros(d, dtype=np.float32), 0.0)]
            
            for l in range(L):
                C_l = codebooks[l]
                K_l = C_l.shape[0]
                new_paths = []
                
                for path_codes, recon_so_far, _ in active_paths:
                    residual = x - recon_so_far
                    dists = np.sum((residual - C_l) ** 2, axis=1)
                    
                    k_actual = min(topk, K_l)
                    topk_indices = np.argpartition(dists, k_actual - 1)[:k_actual]
                    
                    for j in topk_indices:
                        new_codes = path_codes + [j]
                        new_recon = recon_so_far + C_l[j]
                        new_error = np.sum((x - new_recon) ** 2)
                        new_paths.append((new_codes, new_recon, new_error))
                
                if len(new_paths) > beam_width:
                    new_paths.sort(key=lambda p: p[2])
                    active_paths = new_paths[:beam_width]
                else:
                    active_paths = new_paths
            
            best_path = min(active_paths, key=lambda p: p[2])
            best_codes, best_recon, _ = best_path
            
            for l in range(L):
                codes_all[l, global_i] = best_codes[l]
            recon[global_i] = best_recon
    
    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"  Beam Search MSE: {total_mse:.6f}")
    
    print(f"[Time] encode_topk_beam_search_batch: {time.time() - start_time:.2f}s")
    return codes_all, recon


def encode_topk_collision_aware(X, codebooks, topk=4, beam_width=16, alpha=0.1, batch_size=1000, verbose=False):
    """
    冲突感知的 Top-K Beam Search 编码
    
    核心改进：在选择最终路径时，不仅考虑重构误差(MSE)，还考虑路径的使用频率。
    优先选择"使用较少"的路径，从而分散样本到不同的 SID，降低冲突率。
    
    选择策略：score = MSE + alpha * usage_penalty
    - MSE: 重构误差（越小越好）
    - usage_penalty: 路径使用次数的惩罚（越少越好）
    - alpha: 平衡系数，控制冲突惩罚的强度
    
    Args:
        X: Input data (N, d)
        codebooks: List of L codebooks, each (K, d)
        topk: Number of top candidates to consider at each level
        beam_width: Maximum number of paths to keep
        alpha: Collision penalty weight (0=pure MSE, higher=more diversity)
        batch_size: Number of samples to process together
        verbose: Print detailed info
    
    Returns:
        codes_all: (L, N) integer codes
        recon: Reconstructed data (N, d)
    """
    start_time = time.time()
    n, d = X.shape
    L = len(codebooks)
    X = X.astype(np.float32, copy=False)
    
    if verbose:
        print(f"\n=== Collision-Aware Beam Search Encoding ===")
        print(f"  Samples: {n}, Levels: {L}, TopK: {topk}, BeamWidth: {beam_width}, Alpha: {alpha}")
    
    codes_all = np.empty((L, n), dtype=np.int32)
    recon = np.zeros_like(X)
    
    # 路径使用计数器：key = tuple(path), value = count
    path_usage = defaultdict(int)
    
    # 计算 MSE 的归一化因子（用于平衡 MSE 和 usage_penalty 的量级）
    # 先用一小部分样本估计 MSE 的典型值
    sample_size = min(1000, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    sample_mse = []
    for idx in sample_indices[:100]:  # 快速估计
        x = X[idx]
        r = x.copy()
        for l in range(L):
            C_l = codebooks[l]
            dists = np.sum((r - C_l) ** 2, axis=1)
            best_j = np.argmin(dists)
            r = r - C_l[best_j]
        sample_mse.append(np.sum(r ** 2))
    mse_scale = np.mean(sample_mse) if sample_mse else 1.0
    
    if verbose:
        print(f"  MSE scale (for normalization): {mse_scale:.4f}")
    
    # Process in batches
    num_batches = (n + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Collision-Aware Encoding"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n)
        X_batch = X[start_idx:end_idx]
        B = X_batch.shape[0]
        
        for local_i in range(B):
            global_i = start_idx + local_i
            x = X_batch[local_i]
            
            active_paths = [([], np.zeros(d, dtype=np.float32), 0.0)]
            
            for l in range(L):
                C_l = codebooks[l]
                K_l = C_l.shape[0]
                new_paths = []
                
                for path_codes, recon_so_far, _ in active_paths:
                    residual = x - recon_so_far
                    dists = np.sum((residual - C_l) ** 2, axis=1)
                    
                    k_actual = min(topk, K_l)
                    topk_indices = np.argpartition(dists, k_actual - 1)[:k_actual]
                    
                    for j in topk_indices:
                        new_codes = path_codes + [j]
                        new_recon = recon_so_far + C_l[j]
                        new_error = np.sum((x - new_recon) ** 2)
                        new_paths.append((new_codes, new_recon, new_error))
                
                if len(new_paths) > beam_width:
                    new_paths.sort(key=lambda p: p[2])
                    active_paths = new_paths[:beam_width]
                else:
                    active_paths = new_paths
            
            # 冲突感知的路径选择
            # score = normalized_mse + alpha * log(1 + usage_count)
            def compute_score(path):
                codes, _, mse = path
                path_key = tuple(codes)
                usage = path_usage[path_key]
                # 使用 log 来平滑惩罚，避免过度惩罚高频路径
                usage_penalty = np.log1p(usage)  # log(1 + usage)
                # 归一化 MSE
                normalized_mse = mse / mse_scale
                return normalized_mse + alpha * usage_penalty
            
            best_path = min(active_paths, key=compute_score)
            best_codes, best_recon, _ = best_path
            
            # 更新路径使用计数
            path_key = tuple(best_codes)
            path_usage[path_key] += 1
            
            for l in range(L):
                codes_all[l, global_i] = best_codes[l]
            recon[global_i] = best_recon
    
    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"  Collision-Aware MSE: {total_mse:.6f}")
        print(f"  Unique paths used: {len(path_usage)}")
        # 统计路径使用分布
        usage_counts = list(path_usage.values())
        print(f"  Path usage: min={min(usage_counts)}, max={max(usage_counts)}, mean={np.mean(usage_counts):.1f}")
    
    print(f"[Time] encode_topk_collision_aware: {time.time() - start_time:.2f}s")
    return codes_all, recon


def encode_greedy(X, codebooks, verbose=False):
    """
    Standard greedy encoding (original RQ behavior).
    
    At each level, selects the nearest centroid for the current residual.
    This is the baseline for comparison with beam search.
    
    Args:
        X: Input data (N, d)
        codebooks: List of L codebooks
        verbose: Print detailed info
    
    Returns:
        codes_all: (L, N) integer codes
        recon: Reconstructed data (N, d)
    """
    start_time = time.time()
    n, d = X.shape
    L = len(codebooks)
    X = X.astype(np.float32, copy=False)
    
    codes_all = np.empty((L, n), dtype=np.int32)
    R = X.copy()  # residual
    recon = np.zeros_like(X)
    
    for l in range(L):
        C_l = codebooks[l]  # (K, d)
        # Compute distances: (N, K)
        # ||r - c||^2 = ||r||^2 - 2*r@c.T + ||c||^2
        r_sq = np.sum(R ** 2, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(C_l ** 2, axis=1, keepdims=True).T  # (1, K)
        dists = r_sq - 2 * (R @ C_l.T) + c_sq  # (N, K)
        
        codes_l = np.argmin(dists, axis=1)  # (N,)
        codes_all[l] = codes_l
        
        # Update residual
        selected = C_l[codes_l]  # (N, d)
        R -= selected
        recon += selected
    
    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"  Greedy MSE: {total_mse:.6f}")
    
    print(f"[Time] encode_greedy: {time.time() - start_time:.2f}s")
    return codes_all, recon


def compare_encoding_methods(X, codebooks, topk=4, beam_width=16, verbose=True):
    """
    Compare greedy vs beam search encoding.
    
    Returns:
        dict with comparison metrics
    """
    print("\n" + "=" * 60)
    print("Comparing Encoding Methods")
    print("=" * 60)
    
    # Greedy encoding
    codes_greedy, recon_greedy = encode_greedy(X, codebooks, verbose=verbose)
    mse_greedy = np.mean((X - recon_greedy) ** 2)
    
    # Beam search encoding
    codes_beam, recon_beam = encode_topk_beam_search_batch(
        X, codebooks, topk=topk, beam_width=beam_width, verbose=verbose
    )
    mse_beam = np.mean((X - recon_beam) ** 2)
    
    # Compare collision rates
    def collision_rate(codes):
        combos = len(set(map(tuple, codes.T)))
        return 1 - combos / codes.shape[1]
    
    collision_greedy = collision_rate(codes_greedy)
    collision_beam = collision_rate(codes_beam)
    
    # How many samples changed their path?
    changed = np.any(codes_greedy != codes_beam, axis=0)
    changed_ratio = np.mean(changed)
    
    print(f"\n--- Comparison Results ---")
    print(f"  Greedy MSE:     {mse_greedy:.6f}")
    print(f"  Beam MSE:       {mse_beam:.6f}")
    print(f"  MSE Improvement: {(mse_greedy - mse_beam) / mse_greedy * 100:.2f}%")
    print(f"  Greedy Collision Rate: {collision_greedy:.4f}")
    print(f"  Beam Collision Rate:   {collision_beam:.4f}")
    print(f"  Samples with changed path: {changed_ratio * 100:.2f}%")
    
    return {
        'mse_greedy': mse_greedy,
        'mse_beam': mse_beam,
        'collision_greedy': collision_greedy,
        'collision_beam': collision_beam,
        'changed_ratio': changed_ratio,
        'codes_greedy': codes_greedy,
        'codes_beam': codes_beam
    }


def deal_with_deduplicate(df):
    """Handle duplicates by appending row index"""
    df_with_index = df.with_row_index()

    result_df = df_with_index.with_columns(
        pl.when(pl.len().over("codes") > 1)
        .then(
            pl.col("codes").list.concat(
                pl.col("index").rank(method="ordinal").over("codes").cast(pl.Int64)
            )
        )
        .otherwise(pl.col("codes"))
        .alias("codes")
    ).drop("index")

    return result_df


def analyze_codes(codes, title="", verbose=True):
    """Analyze code distribution and collision rate"""
    N, M = codes.shape
    if verbose:
        if title:
            print(f"\n{title}")
        print(f"  Total items: {N}")
        for l in range(M):
            unique_count = len(np.unique(codes[:, l]))
            print(f"  Level {l+1}: unique codes = {unique_count}")

        # Check collision rate
        combos = len(set(map(tuple, codes)))
        collision_rate = 1 - combos / N
        print(f"  Unique full-paths: {combos}")
        print(f"  Collision rate: {collision_rate:.4f}")
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Constrained RQ-KMeans clustering")
    parser.add_argument('--root', type=str, default="./yambda/sequential-multievent-500m", help="Root directory for data")
    parser.add_argument("--dataset", type=str, default="sequential-multievent-500m.item_emb.npy", help="Dataset name (e.g., Industrial_and_Scientific)")
    parser.add_argument("--k", type=int, default=512, help="Number of clusters per level")
    parser.add_argument("--l", type=int, default=4, help="Number of levels")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--topk", type=int, default=8, help="Top-K candidates per level for beam search")
    parser.add_argument("--beam_width", type=int, default=32, help="Beam width for path search")
    parser.add_argument("--use_beam_search", action="store_true", help="Use beam search encoding instead of greedy")
    parser.add_argument("--use_collision_aware", action="store_true", help="Use collision-aware beam search encoding")
    parser.add_argument("--alpha", type=float, default=1.0, help="Collision penalty weight for collision-aware encoding")
    parser.add_argument("--compare_methods", action="store_true", help="Compare greedy vs beam search encoding")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check k-means-constrained availability
    if not HAS_CONSTRAINED:
        print("Error: k-means-constrained is required but not installed")
        print("Install with: pip install k-means-constrained")
        exit(1)

    # Load data
    print("=" * 60)
    print(f"RQ-KMeans Constrained Training")
    print("=" * 60)

    t0 = time.time()
    print("root: ", args.root)
    print("dataset: ", args.dataset)
    data_path = os.path.join(args.root, args.dataset)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        exit(1)

    embeddings = np.load(data_path).astype(np.float32)
    print(f"Loaded embeddings from: {data_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"[Time: {time.time()-t0:.2f}s]\n")

    K_values = [args.k] * args.l

    # Run residual K-means
    t1 = time.time()
    codes_all, codebooks, recon = residual_kmeans_constrained(
        embeddings, K=K_values, L=args.l, random_state=args.seed,
        verbose=args.verbose, max_iter=args.max_iter
    )
    print(f"\n[Time] Total training time: {time.time()-t1:.2f}s")

    # Analyze codes (from training - greedy)
    analyze_codes(codes_all.T, title="Code Statistics (Training/Greedy):", verbose=True)

    # Save codebooks
    output_dir = args.root
    os.makedirs(output_dir, exist_ok=True)

    t2 = time.time()
    codebook_path = os.path.join(output_dir, f'{args.dataset}.codebooks_constrained.npz')
    np.savez_compressed(codebook_path,
                       **{f'codebook_{i}': cb for i, cb in enumerate(codebooks)})
    print(f"\n[Time] Saved codebooks: {time.time()-t2:.2f}s")
    print(f"Codebooks saved to: {codebook_path}")

    # ========================================
    # Encoding Phase: Greedy vs Beam Search
    # ========================================
    
    # Decide which encoding to use for final SID construction
    if args.compare_methods:
        # Compare both methods
        comparison = compare_encoding_methods(
            embeddings, codebooks, 
            topk=args.topk, beam_width=args.beam_width, 
            verbose=args.verbose
        )
        # Use beam search codes if it's better
        if comparison['mse_beam'] < comparison['mse_greedy']:
            print("\n>>> Using Beam Search encoding for final SID (better MSE)")
            codes_all = comparison['codes_beam']
            # Reconstruct from beam search codes
            recon = np.zeros_like(embeddings)
            for l in range(args.l):
                recon += codebooks[l][codes_all[l]]
        else:
            print("\n>>> Using Greedy encoding for final SID")
            # codes_all already from training
    elif args.use_collision_aware:
        # Use collision-aware beam search encoding
        print("\n" + "=" * 60)
        print("Re-encoding with Collision-Aware Beam Search")
        print("=" * 60)
        codes_all, recon = encode_topk_collision_aware(
            embeddings, codebooks,
            topk=args.topk, beam_width=args.beam_width,
            alpha=args.alpha,
            verbose=args.verbose
        )
        analyze_codes(codes_all.T, title="Code Statistics (Collision-Aware):", verbose=True)
    elif args.use_beam_search:
        # Use beam search encoding
        print("\n" + "=" * 60)
        print("Re-encoding with Top-K Beam Search")
        print("=" * 60)
        codes_all, recon = encode_topk_beam_search_batch(
            embeddings, codebooks,
            topk=args.topk, beam_width=args.beam_width,
            verbose=args.verbose
        )
        analyze_codes(codes_all.T, title="Code Statistics (Beam Search):", verbose=True)
    # else: use codes_all from training (greedy)

    # Prepare codes (+1 offset for token format)
    codes_plus_one = codes_all.T + 1
    codes_df = pl.DataFrame({'codes': [list(c) for c in codes_plus_one]})

    # Deduplication
    t4 = time.time()
    codes_dedup = deal_with_deduplicate(codes_df)
    print(f"[Time] Deduplication: {time.time()-t4:.2f}s")

    # Save original codes (not deduplicated)
    codes_path = os.path.join(output_dir, f'{args.dataset}.codes_constrained.npy')
    np.save(codes_path, codes_all.T)
    print(f"Codes saved to: {codes_path}")

    # Generate JSON index
    t5 = time.time()
    codes_json = {}
    for id, row in tqdm(enumerate(codes_dedup.iter_rows(named=True)), total=len(codes_dedup), desc="Generating JSON index"):
        codes_ = []
        for i, code in enumerate(row['codes']):
            codes_.append(f'<|{chr(97+i)}_{code}|>')
        codes_json[str(id)] = codes_

    # Save JSON index
    json_path = os.path.join(output_dir, f'{args.dataset}.index.json')
    with open(json_path, 'w') as f:
        json.dump(codes_json, f, indent=2)
    print(f"[Time] JSON index generation: {time.time()-t5:.2f}s")
    print(f"JSON index saved to: {json_path}")

    # Print final statistics
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    print(f"- Original data shape: {embeddings.shape}")
    print(f"- Number of levels: {args.l}")
    print(f"- K values per level: {K_values}")
    print(f"- Final reconstruction error (MSE): {np.mean((embeddings - recon) ** 2):.6f}")
    encoding_method = 'Collision-Aware' if args.use_collision_aware else ('Beam Search' if args.use_beam_search else 'Greedy')
    print(f"- Encoding method: {encoding_method}")
    if args.use_beam_search or args.use_collision_aware:
        print(f"- TopK: {args.topk}, BeamWidth: {args.beam_width}")
    if args.use_collision_aware:
        print(f"- Alpha (collision penalty): {args.alpha}")

    # Deduplication statistics
    codes_str = codes_df.with_columns(
        pl.col("codes").map_elements(lambda x: ','.join(map(str, x)), return_dtype=pl.Utf8).alias("codes_str")
    )
    duplicates = (codes_str
                  .group_by("codes_str")
                  .count()
                  .filter(pl.col("count") > 1)
                  .sort("count", descending=True))

    if len(duplicates) > 0:
        print(f"\nDeduplication Statistics:")
        print(f"- Number of duplicate groups: {len(duplicates)}")
        print(f"- Total duplicates: {duplicates['count'].sum() - len(duplicates)}")
        print(f"- Largest duplicate group size: {duplicates['count'].max()}")
    else:
        print("\nNo duplicates found!")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
