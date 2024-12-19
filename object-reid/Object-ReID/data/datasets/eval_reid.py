# encoding: utf-8

import numpy as np
import torch


def eval_func_me(qf, gf, q_pids, g_pids, q_camids, g_camids, q_classes, g_classes, max_rank=50, block_size=2048, logger=None):
    """
    Function to evaluate ReID results without loading the entire distance matrix into memory.

    Args:
    - qf (tensor): Normalized Query features
    - gf (tensor): Normalized Galery features
    - q_pids (array): Query IDs
    - g_pids (array): Galery IDs
    - q_camids (array): Query cam IDs
    - g_camids (array): Galery cam IDs
    - q_classes (array): Query object classes
    - g_classes (array): Galery object classes
    - max_rank (int): Maximum k to calculate for CMC-k
    - block_size (int): Number of queries to process at a time; distance matrix size = block_size x galery size
    - logger (logger): Logger to keep track of progress
    """
    m, n = qf.shape[0], gf.shape[0]
    num_q, num_g = m, n
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    avg_dists_match = []
    avg_dists_mismatch = []
    avg_dists_mismatch_class = []
    quantiles_match = []
    quantiles_mismatch = []
    first_misses = []
    last_hits = []
    num_valid_q = 0.  # number of valid query
    # process queries in blocks
    for j in range(0, num_q, block_size):
        if j+block_size > num_q:
            k = num_q
        else:
            k = j + block_size
        q_part = qf[j:k]
        mp = q_part.shape[0]
        # I only understand this calculation as long as the features are normalized (which they should be!)
        # In that case this entire line results in a matrix filled with constant 2.0
        distmat_ = torch.pow(q_part, 2).sum(dim=1, keepdim=True).expand(mp, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, mp).t()
        # Then this is 1*distmat_ - 2*(q_part @ gf.T) = 1*2 - 2*(q_part @ gf.T) = 2 - 2*cos(D)
        # The cosine distance in interval [0, 4]
        distmat_.addmm_(1, -2, q_part, gf.t())
        distmat = distmat_.cpu().numpy()
        del distmat_
        torch.cuda.empty_cache()
        # sort galery samples by distance to query
        indices = np.argsort(distmat, axis=1)
        # True where query ID and galery ID match
        matches = (g_pids[indices] == q_pids[j:k, np.newaxis]).astype(np.int32)
        # distmat sorted by distance
        distmat_sorted = np.take_along_axis(distmat, indices, axis=1)
        # True where query object class and galery object class match
        class_matches = (g_classes[indices] == q_classes[j:k, np.newaxis]).astype(np.int32)
        del distmat

        # avg distances per query to matches
        match_dists = np.where(matches, distmat_sorted, 0).sum(axis=1) / matches.sum(axis=1)
        # avg distances per query to mismatches
        mismatch_dists = np.where(matches == False, distmat_sorted, 0).sum(axis=1) / (matches == False).sum(axis=1)
        # avg distances per query to mismatches within object class
        mismatch_class_dists = np.where((matches == False) == class_matches, distmat_sorted, 0).sum(axis=1) / ((matches == False) == class_matches).sum(axis=1)
        avg_dists_match.append(match_dists)
        avg_dists_mismatch.append(mismatch_dists)
        avg_dists_mismatch_class.append(mismatch_class_dists)

        # quantiles distances per query
        match_dists = np.sort(np.where(matches, distmat_sorted, -1), axis=1)
        match_start_inds = match_dists.shape[1] - 1 - np.argmin(match_dists[:, ::-1], axis=1)
        quantiles_match_ = []
        for q in [0.25, 0.5, 0.75]:
            inds = ((match_dists.shape[1] - match_start_inds) * q + match_start_inds).astype(np.int32)
            quantiles_match_.append(match_dists[np.arange(inds.shape[0]), inds].copy())
        quantiles_match.append(np.stack(quantiles_match_))
        del match_dists
        mismatch_dists = np.sort(np.where(matches == False, distmat_sorted, -1), axis=1)
        mismatch_start_inds = mismatch_dists.shape[1] - 1 - np.argmin(mismatch_dists[:, ::-1], axis=1)
        quantiles_mismatch_ = []
        for q in [0.25, 0.5, 0.75]:
            inds = ((mismatch_dists.shape[1] - mismatch_start_inds) * q + mismatch_start_inds).astype(np.int32)
            quantiles_mismatch_.append(mismatch_dists[np.arange(inds.shape[0]), inds].copy())
        quantiles_mismatch.append(np.stack(quantiles_mismatch_))
        del mismatch_dists
        del distmat_sorted

        # first miss + last hit per query
        first_miss = np.argmin(matches, axis=1)
        first_misses.append(first_miss)
        last_hit = matches.shape[1] - 1 - np.argmax(matches[:, ::-1], axis=1)
        last_hits.append(last_hit)

        # get query pid and camid
        q_pids_ = q_pids[j:k]
        q_camids_ = q_camids[j:k]

        # remove gallery samples that have the same pid and camid with query
        # how does this code work? I have no idea!
        remove = (g_pids[indices] == np.tile(q_pids_, (g_pids.shape[0], 1)).T) & (g_camids[indices] == np.tile(q_camids_, (g_camids.shape[0], 1)).T)
        keep = np.invert(remove)
        del indices

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = np.where(keep, matches, 0)
        # this condition is true when query identity appears in gallery
        valids = np.any(orig_cmc, axis=1)
        orig_cmc = orig_cmc[valids]

        cmc = orig_cmc.cumsum(axis=1)
        # last index where orig_cmc == 1
        max_pos_idx = orig_cmc.shape[1] - 1 - np.argmax(orig_cmc[:, ::-1], axis=1)
        inp = cmc[np.arange(max_pos_idx.shape[0]), max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp.copy())

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:, :max_rank].copy())
        num_valid_q += cmc.shape[0]
        del cmc

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum(axis=1)
        tmp_cmc = orig_cmc.cumsum(axis=1)
        tmp_cmc = tmp_cmc / np.tile(np.arange(tmp_cmc.shape[1]) + 1.0, (tmp_cmc.shape[0], 1))
        tmp_cmc = tmp_cmc * orig_cmc
        AP = tmp_cmc.sum(axis=1) / num_rel
        all_AP.append(AP.copy())
        del tmp_cmc
        if logger is not None:
            logger.info(f"Evaluation Metric Calculation {k}/{num_q} done.")

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.concatenate(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(np.concatenate(all_AP))
    mINP = np.mean(np.concatenate(all_INP))
    quantiles_match = np.concatenate(quantiles_match, axis=1)
    quantiles_mismatch = np.concatenate(quantiles_mismatch, axis=1)
    tabular_data = [
        np.concatenate(avg_dists_match),
        quantiles_match[0],
        quantiles_match[1],
        quantiles_match[2],
        np.concatenate(avg_dists_mismatch_class),
        np.concatenate(avg_dists_mismatch),
        quantiles_mismatch[0],
        quantiles_mismatch[1],
        quantiles_mismatch[2],
        np.concatenate(first_misses),
        np.concatenate(last_hits),
        np.concatenate(all_AP)
    ]
    # Distances are scaled to be in [0, 180] instead of [0, 4]
    # to be more analogous to degrees. This is kinda misleading.
    for i, t in enumerate(tabular_data[:-2]):
        tabular_data[i] = t * 45

    return all_cmc, mAP, mINP, tabular_data
