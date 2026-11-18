import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

@torch.no_grad()
def compute_batch_metrics(outputs, targets, active_mask, loc_embeds, projected_bert_embeds, align_embeds, device):
    """
    Evaluates execution step error states across both sub-task objectives.
    
    Returns:
        prox_error (float): Proximity classification error scalar value.
        clip_error (float): Cross-modal location retrieval error scalar value.
    """
    # 1. Proximity Classification Error
    preds = (outputs >= 0.0).float()
    active_errors = (preds != (targets >= 1.0).float()).float() * active_mask
    batch_prox_error = active_errors.sum() / torch.clamp(active_mask.sum(), min=1.0)
    
    # 2. CLIP Text-to-Location Retrieval Error
    norm_loc_feats = nn.functional.normalize(loc_embeds, dim=-1)
    norm_bert_feats = nn.functional.normalize(projected_bert_embeds, dim=-1)
    
    similarity_matrix = torch.matmul(norm_bert_feats, norm_loc_feats.T)
    closest_loc_indices = torch.argmax(similarity_matrix, dim=-1)
    
    # Create [B, B] mask where mask[i, j] is True if text i matches text j
    valid_match_mask = torch.isclose(
        align_embeds.unsqueeze(0),  # [1, B, D]
        align_embeds.unsqueeze(1),  # [B, 1, D]
        atol=1e-5
    ).all(dim=-1)
    
    # Check if the retrieved location index is a valid text match for each query row
    current_align_batch_size = loc_embeds.size(0)
    retrieval_success = valid_match_mask[torch.arange(current_align_batch_size, device=device), closest_loc_indices]
    batch_clip_error = 1.0 - retrieval_success.float().mean()
    
    return batch_prox_error.item(), batch_clip_error.item()

def _haversine_km(a, b):
    """Great-circle km between lat/lon pairs. a, b: (..., 2) degrees."""
    R = 6371.0088
    lat1, lon1 = torch.deg2rad(a[..., 0]), torch.deg2rad(a[..., 1])
    lat2, lon2 = torch.deg2rad(b[..., 0]), torch.deg2rad(b[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (dlat / 2).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2).sin() ** 2
    return 2 * R * torch.asin(h.clamp(0, 1).sqrt())


def evaluate_ranking_metrics(model, dataloader, unique_coords, device, neutral_pad, k_values, seen_mask, task='next_event'):
    """
    Executes unbiased global rank evaluation calculations over data loader iterations.
    Splits ranking performance metrics into global, seen, and unseen POI subsets.

    Args:
        seen_mask (torch.Tensor): Boolean tensor of shape (num_unique,) on device
                                  where True = POI was seen during training.
        task (str): 'next_event' evaluates every step; 'last_event' evaluates only the
                    final valid step per trajectory.
    """
    model.eval()
    seen_mask = seen_mask.to(device)

    total_recall_hits   = {k: 0 for k in k_values}; total_mrr_sum   = 0.0; total_valid_predictions   = 0
    seen_recall_hits    = {k: 0 for k in k_values}; seen_mrr_sum    = 0.0; seen_valid_predictions    = 0
    unseen_recall_hits  = {k: 0 for k in k_values}; unseen_mrr_sum  = 0.0; unseen_valid_predictions  = 0
    novel_recall_hits   = {k: 0 for k in k_values}; novel_mrr_sum   = 0.0; novel_valid_predictions   = 0
    last_recall_hits    = {k: 0 for k in k_values}; last_mrr_sum    = 0.0; last_valid_predictions    = 0
    fv_unseen_recall_hits = {k: 0 for k in k_values}; fv_unseen_mrr_sum = 0.0; fv_unseen_valid = 0  # First-View unseen
    rv_unseen_recall_hits = {k: 0 for k in k_values}; rv_unseen_mrr_sum = 0.0; rv_unseen_valid = 0  # Repeat-View unseen
    K_max = max(k_values)
    dist_at_k = {k: [] for k in k_values}

    with torch.no_grad():
        candidate_embs = model.location_encoder(unique_coords)
        unique_embs = F.normalize(candidate_embs, dim=-1)

        for batch_data in tqdm(dataloader, desc="Evaluating"):
            batch_coords  = batch_data['coords'].to(device)
            batch_targets = batch_data['targets'].to(device)
            lengths       = batch_data['lengths'].to(device)

            inputs = batch_coords[:, :-1, :].clone()
            targets = batch_targets[:, 1:]
            adjusted_lengths = lengths - 1

            pad_mask = (inputs[:, :, 0] == -100.0)
            inputs[pad_mask] = neutral_pad

            input_embs = model.location_encoder(inputs)
            timestamps = batch_data['timestamps'].to(device)[:, :-1] if model.use_time_fusion else None
            hidden_states = model(lengths=adjusted_lengths, embeddings=input_embs, timestamps=timestamps)
            hidden_states_norm = F.normalize(hidden_states, dim=-1)

            B, S = targets.shape

            _C = 4096
            num_unique_local = unique_embs.shape[0]
            full_valid_mask = torch.arange(S, device=device).unsqueeze(0) < adjusted_lengths.unsqueeze(1)
            last_ids = batch_targets[torch.arange(B, device=device), adjusted_lengths]  # (B,)

            if task == 'last_event':
                targets = torch.where(full_valid_mask, last_ids.unsqueeze(1).expand(B, S), targets)

            gt_idx = targets.clamp(min=0)
            true_scores = (hidden_states_norm * unique_embs[gt_idx]).sum(-1, keepdim=True)
            greater_counts = torch.zeros(B, S, dtype=torch.long, device=device)
            equal_counts   = torch.zeros(B, S, dtype=torch.long, device=device)
            # Separate rank for last-event prediction at all valid context lengths
            last_gt_idx = torch.where(full_valid_mask, last_ids.unsqueeze(1).expand(B, S), gt_idx)
            true_last_scores = (hidden_states_norm * unique_embs[last_gt_idx]).sum(-1, keepdim=True)
            last_greater = torch.zeros(B, S, dtype=torch.long, device=device)
            last_equal   = torch.zeros(B, S, dtype=torch.long, device=device)
            top_vals = torch.full((B, S, 0), float('-inf'), device=device)
            top_idx  = torch.zeros(B, S, 0, dtype=torch.long, device=device)
            for cs in range(0, num_unique_local, _C):
                ce = min(cs + _C, num_unique_local)
                chunk = torch.matmul(hidden_states_norm, unique_embs[cs:ce].T)
                greater_counts += (chunk > true_scores).sum(dim=-1)
                equal_counts   += (chunk == true_scores).sum(dim=-1)
                last_greater   += (chunk > true_last_scores).sum(dim=-1)
                last_equal     += (chunk == true_last_scores).sum(dim=-1)
                v, i = chunk.topk(min(K_max, chunk.size(-1)), dim=-1)
                top_vals = torch.cat([top_vals, v], dim=-1)
                top_idx  = torch.cat([top_idx, i + cs], dim=-1)
                keep_k = min(K_max, top_vals.size(-1))
                top_vals, keep = top_vals.topk(keep_k, dim=-1)
                top_idx = top_idx.gather(-1, keep)
            ranks = (greater_counts + equal_counts).clamp(min=1)
            last_event_ranks = (last_greater + last_equal).clamp(min=1)

            # Distance@k: km from the true next coordinate to the nearest top-k candidate
            true_next_coords = batch_coords[:, 1:, :]
            cand_coords = unique_coords[top_idx]                          # (B, S, K, 2)
            dists = _haversine_km(true_next_coords.unsqueeze(2), cand_coords)  # (B, S, K)
            for k in k_values:
                dmin = dists[..., :k].min(dim=-1).values[full_valid_mask]
                dist_at_k[k].append(dmin.detach().cpu())

            is_seen_target = seen_mask[targets.clamp(min=0)]

            hist_src = batch_targets[:, :-1]
            match = (hist_src.unsqueeze(2) == targets.unsqueeze(1))
            causal = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool))
            is_repeat = (match & causal.unsqueeze(0)).any(dim=1)
            is_novel_target = ~is_repeat

            valid_mask = full_valid_mask
            valid_ranks = ranks[valid_mask]
            valid_seen  = is_seen_target[valid_mask]
            valid_novel = is_novel_target[valid_mask]

            if valid_ranks.size(0) > 0:
                total_valid_predictions += valid_ranks.size(0)
                total_mrr_sum += (1.0 / valid_ranks.float()).sum().item()
                for k in k_values:
                    total_recall_hits[k] += (valid_ranks <= k).sum().item()

                seen_ranks = valid_ranks[valid_seen]
                if seen_ranks.size(0) > 0:
                    seen_valid_predictions += seen_ranks.size(0)
                    seen_mrr_sum += (1.0 / seen_ranks.float()).sum().item()
                    for k in k_values:
                        seen_recall_hits[k] += (seen_ranks <= k).sum().item()

                unseen_ranks = valid_ranks[~valid_seen]
                if unseen_ranks.size(0) > 0:
                    unseen_valid_predictions += unseen_ranks.size(0)
                    unseen_mrr_sum += (1.0 / unseen_ranks.float()).sum().item()
                    for k in k_values:
                        unseen_recall_hits[k] += (unseen_ranks <= k).sum().item()

                novel_ranks = valid_ranks[valid_novel]
                if novel_ranks.size(0) > 0:
                    novel_valid_predictions += novel_ranks.size(0)
                    novel_mrr_sum += (1.0 / novel_ranks.float()).sum().item()
                    for k in k_values:
                        novel_recall_hits[k] += (novel_ranks <= k).sum().item()

                # Recomputed unseen: ground truth absent from BOTH training and the
                # input context. First-View = absent from both (prior is blind).
                # Repeat-View = masked from training but present in the user's input.
                fv_mask = (~valid_seen) & valid_novel
                rv_mask = (~valid_seen) & (~valid_novel)
                fv_ranks = valid_ranks[fv_mask]
                if fv_ranks.size(0) > 0:
                    fv_unseen_valid += fv_ranks.size(0)
                    fv_unseen_mrr_sum += (1.0 / fv_ranks.float()).sum().item()
                    for k in k_values:
                        fv_unseen_recall_hits[k] += (fv_ranks <= k).sum().item()
                rv_ranks = valid_ranks[rv_mask]
                if rv_ranks.size(0) > 0:
                    rv_unseen_valid += rv_ranks.size(0)
                    rv_unseen_mrr_sum += (1.0 / rv_ranks.float()).sum().item()
                    for k in k_values:
                        rv_unseen_recall_hits[k] += (rv_ranks <= k).sum().item()

            # last_stats: last-event prediction at all valid context lengths
            last_ranks = last_event_ranks[full_valid_mask]
            if last_ranks.size(0) > 0:
                last_valid_predictions += last_ranks.size(0)
                last_mrr_sum += (1.0 / last_ranks.float()).sum().item()
                for k in k_values:
                    last_recall_hits[k] += (last_ranks <= k).sum().item()

    def _stats(hits, mrr, n):
        return {"recall": {k: hits[k] / n if n > 0 else 0.0 for k in k_values},
                "mrr": mrr / n if n > 0 else 0.0, "count": n}

    global_stats = _stats(total_recall_hits, total_mrr_sum, total_valid_predictions)
    for k in k_values:
        vals = torch.cat(dist_at_k[k]) if dist_at_k[k] else torch.tensor([])
        global_stats.setdefault("dist_mean", {})[k] = vals.mean().item() if vals.numel() else 0.0
        global_stats.setdefault("dist_median", {})[k] = vals.median().item() if vals.numel() else 0.0

    return (
        global_stats,
        _stats(seen_recall_hits,   seen_mrr_sum,   seen_valid_predictions),
        _stats(unseen_recall_hits, unseen_mrr_sum, unseen_valid_predictions),
        _stats(novel_recall_hits,  novel_mrr_sum,  novel_valid_predictions),
        _stats(last_recall_hits,   last_mrr_sum,   last_valid_predictions),
        _stats(fv_unseen_recall_hits, fv_unseen_mrr_sum, fv_unseen_valid),
        _stats(rv_unseen_recall_hits, rv_unseen_mrr_sum, rv_unseen_valid),
    )


@torch.no_grad()
def evaluate_coarse_ranking_metrics(
    model, dataloader, unique_coords, device, neutral_pad, k_values, seen_mask,
    task='next_event',
    poi_h3_cell_idx=None,
    h3_centroids=None,
    poi_region_indices=None,
    region_centroids=None,
):
    """
    Coarse-grained recall: ranks H3 cells and/or OSM regions by centroid embedding
    similarity to the query, then checks if the ground-truth POI's cell/region is
    in the top-K.

    Args:
        poi_h3_cell_idx:  LongTensor (N,) — H3 cell index per POI (from dataset.unique_coord_h3_indices).
                          Pass None to skip H3 evaluation.
        h3_centroids:     FloatTensor (C, 2) — H3 cell centroid (lat, lon) coordinates.
        poi_region_indices: LongTensor (N, max_R) — region IDs per POI, -1 padded
                            (from dataset.poi_region_indices). Pass None to skip region eval.
        region_centroids: FloatTensor (R, 2) — OSM region centroid coordinates.

    Returns 5-tuple matching evaluate_ranking_metrics. Each stats dict gains optional
    keys 'h3_recall' and 'region_recall': {k: float} when the corresponding inputs are provided.
    """
    model.eval()
    seen_mask = seen_mask.to(device)
    do_h3 = poi_h3_cell_idx is not None and h3_centroids is not None
    do_region = poi_region_indices is not None and region_centroids is not None

    # Precompute centroid embeddings
    if do_h3:
        h3_centroids_dev = h3_centroids.to(device)
        h3_centroid_embs = F.normalize(model.location_encoder(h3_centroids_dev), dim=-1)  # (C, D)
        poi_h3_dev = poi_h3_cell_idx.to(device)
    if do_region:
        region_centroids_dev = region_centroids.to(device)
        region_centroid_embs = F.normalize(model.location_encoder(region_centroids_dev), dim=-1)  # (R, D)
        poi_region_dev = poi_region_indices.to(device)  # (N, max_R)

    with torch.no_grad():
        candidate_embs = F.normalize(model.location_encoder(unique_coords.to(device)), dim=-1)

    def _mk():
        return {k: 0 for k in k_values}

    total_hits = _mk(); total_mrr = 0.0; total_n = 0
    seen_hits  = _mk(); seen_mrr  = 0.0; seen_n  = 0
    unseen_hits= _mk(); unseen_mrr= 0.0; unseen_n= 0
    novel_hits = _mk(); novel_mrr = 0.0; novel_n = 0
    last_hits  = _mk(); last_mrr  = 0.0; last_n  = 0

    h3_total_hits = _mk(); h3_total_n = 0
    h3_seen_hits  = _mk(); h3_seen_n  = 0
    h3_unseen_hits= _mk(); h3_unseen_n= 0
    h3_novel_hits = _mk(); h3_novel_n = 0

    reg_total_hits = _mk(); reg_total_n = 0
    reg_seen_hits  = _mk(); reg_seen_n  = 0
    reg_unseen_hits= _mk(); reg_unseen_n= 0
    reg_novel_hits = _mk(); reg_novel_n = 0

    _C = 4096
    num_unique = candidate_embs.shape[0]

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Coarse eval"):
            batch_coords  = batch_data['coords'].to(device)
            batch_targets = batch_data['targets'].to(device)
            lengths       = batch_data['lengths'].to(device)

            inputs = batch_coords[:, :-1, :].clone()
            targets = batch_targets[:, 1:]
            adj_len = lengths - 1

            pad_mask = (inputs[:, :, 0] == -100.0)
            inputs[pad_mask] = neutral_pad

            input_embs = model.location_encoder(inputs)
            timestamps = batch_data['timestamps'].to(device)[:, :-1] if model.use_time_fusion else None
            hidden_states = model(lengths=adj_len, embeddings=input_embs, timestamps=timestamps)
            h_norm = F.normalize(hidden_states, dim=-1)  # (B, S, D)

            B, S = targets.shape
            valid_mask = torch.arange(S, device=device).unsqueeze(0) < adj_len.unsqueeze(1)
            last_ids = batch_targets[torch.arange(B, device=device), adj_len]

            if task == 'last_event':
                targets = torch.where(valid_mask, last_ids.unsqueeze(1).expand(B, S), targets)

            gt_idx = targets.clamp(min=0)

            # ── POI-level rank (chunked, same pattern as evaluate_ranking_metrics) ──
            true_scores = (h_norm * candidate_embs[gt_idx]).sum(-1, keepdim=True)
            greater_counts = torch.zeros(B, S, dtype=torch.long, device=device)
            equal_counts   = torch.zeros(B, S, dtype=torch.long, device=device)
            last_gt_idx = torch.where(valid_mask, last_ids.unsqueeze(1).expand(B, S), gt_idx)
            true_last_scores = (h_norm * candidate_embs[last_gt_idx]).sum(-1, keepdim=True)
            last_greater = torch.zeros(B, S, dtype=torch.long, device=device)
            last_equal   = torch.zeros(B, S, dtype=torch.long, device=device)
            for cs in range(0, num_unique, _C):
                ce = min(cs + _C, num_unique)
                chunk = torch.matmul(h_norm, candidate_embs[cs:ce].T)
                greater_counts += (chunk > true_scores).sum(dim=-1)
                equal_counts   += (chunk == true_scores).sum(dim=-1)
                last_greater   += (chunk > true_last_scores).sum(dim=-1)
                last_equal     += (chunk == true_last_scores).sum(dim=-1)
            poi_ranks = (greater_counts + equal_counts).clamp(min=1)
            last_ranks_poi = (last_greater + last_equal).clamp(min=1)

            is_seen = seen_mask[gt_idx]
            hist_src = batch_targets[:, :-1]
            match = (hist_src.unsqueeze(2) == targets.unsqueeze(1))
            causal = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool))
            is_novel = ~(match & causal.unsqueeze(0)).any(dim=1)

            vr = poi_ranks[valid_mask]
            vs = is_seen[valid_mask]
            vn = is_novel[valid_mask]
            if vr.numel() > 0:
                total_n += vr.numel(); total_mrr += (1./vr.float()).sum().item()
                for k in k_values: total_hits[k] += (vr <= k).sum().item()
                sr = vr[vs]; seen_n += sr.numel(); seen_mrr += (1./sr.float()).sum().item() if sr.numel() else 0
                for k in k_values: seen_hits[k] += (sr <= k).sum().item()
                ur = vr[~vs]; unseen_n += ur.numel(); unseen_mrr += (1./ur.float()).sum().item() if ur.numel() else 0
                for k in k_values: unseen_hits[k] += (ur <= k).sum().item()
                nr = vr[vn]; novel_n += nr.numel(); novel_mrr += (1./nr.float()).sum().item() if nr.numel() else 0
                for k in k_values: novel_hits[k] += (nr <= k).sum().item()
            lr = last_ranks_poi[valid_mask]
            if lr.numel() > 0:
                last_n += lr.numel(); last_mrr += (1./lr.float()).sum().item()
                for k in k_values: last_hits[k] += (lr <= k).sum().item()

            # ── H3-level rank ──────────────────────────────────────────────────────
            if do_h3:
                cell_sims = h_norm @ h3_centroid_embs.T        # (B, S, C)
                gt_h3_cell = poi_h3_dev[gt_idx]                # (B, S)
                # -1 means the gt POI has no H3 cell in the training vocab → skip
                h3_valid = valid_mask & (gt_h3_cell >= 0)
                gt_cell_sim = cell_sims.gather(-1, gt_h3_cell.clamp(0).unsqueeze(-1)).squeeze(-1)  # (B, S)
                h3_rank = (cell_sims > gt_cell_sim.unsqueeze(-1)).sum(-1) + 1  # (B, S)

                hv = h3_rank[h3_valid]; hs = is_seen[h3_valid]; hn = is_novel[h3_valid]
                if hv.numel() > 0:
                    h3_total_n += hv.numel()
                    for k in k_values: h3_total_hits[k] += (hv <= k).sum().item()
                    sh = hv[hs]; h3_seen_n += sh.numel()
                    for k in k_values: h3_seen_hits[k] += (sh <= k).sum().item()
                    uh = hv[~hs]; h3_unseen_n += uh.numel()
                    for k in k_values: h3_unseen_hits[k] += (uh <= k).sum().item()
                    nh = hv[hn]; h3_novel_n += nh.numel()
                    for k in k_values: h3_novel_hits[k] += (nh <= k).sum().item()

            # ── Region-level rank ──────────────────────────────────────────────────
            if do_region:
                region_sims = h_norm @ region_centroid_embs.T  # (B, S, R)
                gt_reg_ids = poi_region_dev[gt_idx]             # (B, S, max_R)
                valid_reg = gt_reg_ids >= 0                     # (B, S, max_R)
                has_region = valid_reg.any(dim=-1)              # (B, S)
                reg_valid = valid_mask & has_region

                # Gather similarity for each candidate region; mask padding
                gt_reg_sims = region_sims.gather(-1, gt_reg_ids.clamp(0))  # (B, S, max_R)
                gt_reg_sims[~valid_reg] = -float('inf')
                best_gt_reg_sim = gt_reg_sims.max(-1).values    # (B, S)
                reg_rank = (region_sims > best_gt_reg_sim.unsqueeze(-1)).sum(-1) + 1  # (B, S)

                rv = reg_rank[reg_valid]; rs = is_seen[reg_valid]; rn = is_novel[reg_valid]
                if rv.numel() > 0:
                    reg_total_n += rv.numel()
                    for k in k_values: reg_total_hits[k] += (rv <= k).sum().item()
                    sr2 = rv[rs]; reg_seen_n += sr2.numel()
                    for k in k_values: reg_seen_hits[k] += (sr2 <= k).sum().item()
                    ur2 = rv[~rs]; reg_unseen_n += ur2.numel()
                    for k in k_values: reg_unseen_hits[k] += (ur2 <= k).sum().item()
                    nr2 = rv[rn]; reg_novel_n += nr2.numel()
                    for k in k_values: reg_novel_hits[k] += (nr2 <= k).sum().item()

    def _stats(hits, mrr, n, h3_h=None, h3_n=None, reg_h=None, reg_n=None):
        d = {"recall": {k: hits[k] / n if n > 0 else 0.0 for k in k_values},
             "mrr": mrr / n if n > 0 else 0.0, "count": n}
        if h3_h is not None:
            d["h3_recall"] = {k: h3_h[k] / h3_n if h3_n > 0 else 0.0 for k in k_values}
            d["h3_count"] = h3_n
        if reg_h is not None:
            d["region_recall"] = {k: reg_h[k] / reg_n if reg_n > 0 else 0.0 for k in k_values}
            d["region_count"] = reg_n
        return d

    return (
        _stats(total_hits,  total_mrr,  total_n,  h3_total_hits,  h3_total_n,  reg_total_hits,  reg_total_n),
        _stats(seen_hits,   seen_mrr,   seen_n,   h3_seen_hits,   h3_seen_n,   reg_seen_hits,   reg_seen_n),
        _stats(unseen_hits, unseen_mrr, unseen_n, h3_unseen_hits, h3_unseen_n, reg_unseen_hits, reg_unseen_n),
        _stats(novel_hits,  novel_mrr,  novel_n,  h3_novel_hits,  h3_novel_n,  reg_novel_hits,  reg_novel_n),
        _stats(last_hits,   last_mrr,   last_n),
    )


def _gt_heuristic_score(inputs, targets, hist_src, valid_hist, unique_coords, log_popularity, gamma, beta):
    """Gravity-history heuristic score for the ground-truth candidate at each step. Returns (B,S)."""
    gt_idx  = targets.clamp(min=0)
    gt_dist = torch.norm(inputs - unique_coords[gt_idx], dim=-1)
    gt_grav = log_popularity[gt_idx] - gamma * torch.log(gt_dist + 1e-5)
    S, device = targets.shape[1], inputs.device
    match   = (hist_src.unsqueeze(1) == gt_idx.unsqueeze(2)).float()  # (B,S,S)
    causal  = torch.tril(torch.ones(S, S, device=device))
    gt_hist = torch.log((match * causal.unsqueeze(0) * valid_hist.unsqueeze(1)).sum(-1) + 1.0)
    return (1.0 - beta) * gt_grav + beta * gt_hist


def _heuristic_chunk(inputs, hist_src, valid_hist, coords_chunk, log_pop_chunk, gamma, beta, cs):
    """Gravity-history heuristic scores for candidates [cs, cs+nc). Returns (B,S,nc)."""
    nc     = coords_chunk.shape[0]
    device = inputs.device
    diff_c = inputs.unsqueeze(2) - coords_chunk.view(1, 1, nc, 2)
    dist_c = torch.norm(diff_c, dim=-1)
    grav_c = log_pop_chunk.view(1, 1, -1) - gamma * torch.log(dist_c + 1e-5)
    cidx   = torch.arange(cs, cs + nc, device=device)
    hm_c   = (hist_src.unsqueeze(-1) == cidx.view(1, 1, -1)).float() * valid_hist.unsqueeze(-1)
    hist_c = torch.log(torch.cumsum(hm_c, dim=1) + 1.0)
    return (1.0 - beta) * grav_c + beta * hist_c


@torch.no_grad()
def evaluate_ensemble_metrics(
    model, dataloader, unique_coords, device, neutral_pad,
    k_values, seen_mask, log_popularity,
    heuristic_gamma, heuristic_beta, heuristic_alpha,
    pad_val_coords=-100.0, pad_val_targets=-100,
    task='next_event',
    poi_h3_cell_idx=None,
    h3_centroids=None,
):
    """
    Single test-set pass evaluating the model + gravity-history ensemble.
    Combined score: (1-α)*zscore(model) + α*zscore(heuristic)
    where heuristic = (1-β)*gravity + β*history.

    When poi_h3_cell_idx and h3_centroids are provided, also computes H3-level
    ensemble recall: z-score(model H3 centroid sims) combined with
    z-score(scatter-max heuristic scores per cell) at the same α weighting.
    H3 recall is stored under the 'h3_recall' key in each returned stats dict.
    """
    model.eval()
    seen_mask = seen_mask.to(device)
    unique_coords_dev = unique_coords.to(device)
    num_unique = unique_coords_dev.shape[0]
    candidate_embs = F.normalize(model.location_encoder(unique_coords_dev), dim=-1)

    do_h3 = poi_h3_cell_idx is not None and h3_centroids is not None
    if do_h3:
        h3_centroid_embs = F.normalize(
            model.location_encoder(h3_centroids.to(device)), dim=-1)  # (C, D)
        poi_h3_dev = poi_h3_cell_idx.to(device)
        num_cells  = h3_centroid_embs.shape[0]

    total_hits = {k: 0 for k in k_values}
    total_mrr = 0.0; total_n = 0
    seen_hits  = {k: 0 for k in k_values}
    seen_mrr   = 0.0; seen_n = 0
    unseen_hits = {k: 0 for k in k_values}
    unseen_mrr  = 0.0; unseen_n = 0

    h3_total_hits  = {k: 0 for k in k_values}; h3_total_n  = 0
    h3_seen_hits   = {k: 0 for k in k_values}; h3_seen_n   = 0
    h3_unseen_hits = {k: 0 for k in k_values}; h3_unseen_n = 0

    _C = 4096  # chunk size over N to avoid (B,S,N) allocations

    for batch_data in tqdm(dataloader, desc="Ensemble eval"):
        batch_coords  = batch_data['coords'].to(device)
        batch_targets = batch_data['targets'].to(device)
        lengths       = batch_data['lengths'].to(device)

        inputs  = batch_coords[:, :-1].clone()
        targets = batch_targets[:, 1:]
        adj_len = lengths - 1
        B, S    = targets.shape

        pad_mask = (inputs[:, :, 0] == pad_val_coords)
        inputs[pad_mask] = neutral_pad

        embs = model.location_encoder(inputs)
        embs[pad_mask] = 0.0
        timestamps = batch_data['timestamps'].to(device)[:, :-1] if model.use_time_fusion else None
        hidden = model(lengths=adj_len, embeddings=embs, timestamps=timestamps)
        h_norm = F.normalize(hidden, dim=-1)  # (B,S,D)

        hist_src   = batch_targets[:, :-1]
        valid_hist = (hist_src != pad_val_targets).float()
        valid_mask = torch.arange(S, device=device).unsqueeze(0) < adj_len.unsqueeze(1)
        if task == 'last_event':
            last_ids = batch_targets[torch.arange(B, device=device), adj_len]  # (B,)
            targets = torch.where(valid_mask, last_ids.unsqueeze(1).expand(B, S), targets)
        is_seen    = seen_mask[targets.clamp(min=0)]

        gt_idx     = targets.clamp(min=0)
        gt_m_score = (h_norm * F.normalize(candidate_embs[gt_idx], dim=-1)).sum(-1)  # (B,S)
        gt_heur    = _gt_heuristic_score(inputs, targets, hist_src, valid_hist,
                                         unique_coords_dev, log_popularity, heuristic_gamma, heuristic_beta)

        # Two-pass chunked scan: pass 1 accumulates z-score stats (and H3 heuristic scores),
        # pass 2 counts POI ranks.
        ranks  = torch.zeros(B, S, dtype=torch.long, device=device)
        m_sum  = torch.zeros(B, S, device=device);  m_sum2 = torch.zeros(B, S, device=device)
        h_sum  = torch.zeros(B, S, device=device);  h_sum2 = torch.zeros(B, S, device=device)

        if do_h3:
            # Running scatter-max: for each H3 cell, keep the highest heuristic score
            # across all POIs that fall in that cell. Initialised to -inf sentinel.
            h3_heur_sims = torch.full((B, S, num_cells), -1e9, device=device)

        for cs in range(0, num_unique, _C):
            ce  = min(cs + _C, num_unique)
            m_c = (h_norm @ candidate_embs[cs:ce].T)
            h_c = _heuristic_chunk(inputs, hist_src, valid_hist,
                                   unique_coords_dev[cs:ce], log_popularity[cs:ce],
                                   heuristic_gamma, heuristic_beta, cs)
            m_sum += m_c.sum(-1);  m_sum2 += (m_c ** 2).sum(-1)
            h_sum += h_c.sum(-1);  h_sum2 += (h_c ** 2).sum(-1)

            if do_h3:
                cell_ids_chunk = poi_h3_dev[cs:ce]          # (nc,)
                valid_c = cell_ids_chunk >= 0
                if valid_c.any():
                    cids = cell_ids_chunk[valid_c].view(1, 1, -1).expand(B, S, -1)
                    h3_heur_sims.scatter_reduce_(
                        2, cids, h_c[:, :, valid_c], reduce='amax', include_self=True)

        m_mean = m_sum / num_unique;  m_std = ((m_sum2 / num_unique - m_mean ** 2).clamp(min=0) + 1e-8).sqrt()
        h_mean = h_sum / num_unique;  h_std = ((h_sum2 / num_unique - h_mean ** 2).clamp(min=0) + 1e-8).sqrt()
        gt_combined = (1.0 - heuristic_alpha) * (gt_m_score - m_mean) / m_std \
                    + heuristic_alpha           * (gt_heur    - h_mean) / h_std

        for cs in range(0, num_unique, _C):
            ce  = min(cs + _C, num_unique)
            m_c = (h_norm @ candidate_embs[cs:ce].T)
            h_c = _heuristic_chunk(inputs, hist_src, valid_hist,
                                   unique_coords_dev[cs:ce], log_popularity[cs:ce],
                                   heuristic_gamma, heuristic_beta, cs)
            combined_c = (1.0 - heuristic_alpha) * (m_c - m_mean.unsqueeze(-1)) / m_std.unsqueeze(-1) \
                       + heuristic_alpha           * (h_c - h_mean.unsqueeze(-1)) / h_std.unsqueeze(-1)
            ranks += (combined_c > gt_combined.unsqueeze(-1)).sum(-1)

        ranks += 1  # 1-indexed

        vr = ranks[valid_mask]
        vs = is_seen[valid_mask]

        if vr.numel() > 0:
            total_n   += vr.numel()
            total_mrr += (1.0 / vr.float()).sum().item()
            for k in k_values:
                total_hits[k] += (vr <= k).sum().item()

            sr = vr[vs]
            if sr.numel() > 0:
                seen_n   += sr.numel()
                seen_mrr += (1.0 / sr.float()).sum().item()
                for k in k_values:
                    seen_hits[k] += (sr <= k).sum().item()

            ur = vr[~vs]
            if ur.numel() > 0:
                unseen_n   += ur.numel()
                unseen_mrr += (1.0 / ur.float()).sum().item()
                for k in k_values:
                    unseen_hits[k] += (ur <= k).sum().item()

        # ── H3 ensemble recall ────────────────────────────────────────────────
        if do_h3:
            h3_model_sims = h_norm @ h3_centroid_embs.T          # (B, S, C)

            # Z-score each component over C cells, then combine at same α
            h3_m_mean = h3_model_sims.mean(-1, keepdim=True)
            h3_m_std  = h3_model_sims.std(-1,  keepdim=True) + 1e-8
            h3_h_mean = h3_heur_sims.mean(-1,  keepdim=True)
            h3_h_std  = h3_heur_sims.std(-1,   keepdim=True) + 1e-8

            h3_combined = (1.0 - heuristic_alpha) * (h3_model_sims - h3_m_mean) / h3_m_std \
                        + heuristic_alpha           * (h3_heur_sims  - h3_h_mean) / h3_h_std

            gt_h3_cell  = poi_h3_dev[gt_idx]                     # (B, S)
            h3_valid    = valid_mask & (gt_h3_cell >= 0)
            gt_h3_score = h3_combined.gather(
                -1, gt_h3_cell.clamp(0).unsqueeze(-1)).squeeze(-1)
            h3_rank = (h3_combined > gt_h3_score.unsqueeze(-1)).sum(-1) + 1  # (B, S)

            hv = h3_rank[h3_valid]; hs = is_seen[h3_valid]
            if hv.numel() > 0:
                h3_total_n += hv.numel()
                for k in k_values: h3_total_hits[k] += (hv <= k).sum().item()
                sh = hv[hs];  h3_seen_n   += sh.numel()
                for k in k_values: h3_seen_hits[k]   += (sh <= k).sum().item()
                uh = hv[~hs]; h3_unseen_n += uh.numel()
                for k in k_values: h3_unseen_hits[k] += (uh <= k).sum().item()

    def _stats(hits, mrr, n, h3_h=None, h3_n=None):
        d = {"recall": {k: hits[k] / n if n > 0 else 0.0 for k in k_values},
             "mrr": mrr / n if n > 0 else 0.0, "count": n}
        if h3_h is not None:
            d["h3_recall"] = {k: h3_h[k] / h3_n if h3_n > 0 else 0.0 for k in k_values}
            d["h3_count"]  = h3_n
        return d

    return (
        _stats(total_hits,  total_mrr,  total_n,  h3_total_hits,  h3_total_n),
        _stats(seen_hits,   seen_mrr,   seen_n,   h3_seen_hits,   h3_seen_n),
        _stats(unseen_hits, unseen_mrr, unseen_n, h3_unseen_hits, h3_unseen_n),
    )


def evaluate_unbiased_metrics(
    model, test_loader, device, num_pois, k_values, poi_coords, args, cat_logits_global=None
):
    """
    Executes unbiased evaluation calculations over testing loader iterations.
    Incorporates an optional category-based tiebreaker for spatial coordinate modes.
    """
    test_recall_hits = {k: 0 for k in k_values}
    test_mrr_sum = 0.0
    test_total_trajectories = 0

    # Ensure the global category mapping tensor is on the correct device
    if cat_logits_global is not None:
        cat_logits_global = cat_logits_global.to(device)

    if model.mode == "latlon":
        global_poi_spatial_embeddings = (
            model.location_encoder(poi_coords.unsqueeze(0)).squeeze(0) 
            if model.location_encoder_type == "rff" 
            else model.coordinate_encoder(model.spatial_batch_norm(poi_coords))
        )

    for batch in test_loader:
        traj_ids, meta, pois, cats, d_ts, d_ss, lats, lons = batch
        meta, pois, cats = meta.to(device), pois.to(device), cats.to(device)
        d_ts, d_ss, lats, lons = d_ts.to(device), d_ss.to(device), lats.to(device), lons.to(device)
        
        logits, _, _, _, cat_logits, predicted_loc_emb, _ = model(
            traj_ids=traj_ids, meta=meta, pois=pois, cats=cats, 
            d_ts=d_ts, d_ss=d_ss, lats=lats, lons=lons, target_pois=pois, global_coords=poi_coords
        )

        prediction_source_indices = model.num_sub_tokens * torch.arange(pois.size(1) - 1, device=device) + model.prediction_offset

        for b in range(pois.size(0)):
            valid_slots = torch.where(pois[b] != model.padding_poi_id)[0]
            if len(valid_slots) <= 1: 
                continue
            
            last_valid_idx = valid_slots[-1].item()
            
            true_target_poi = pois[b, last_valid_idx].item()
            if true_target_poi >= num_pois: 
                continue
            
            true_cat_id = cats[b, last_valid_idx].item()
            
            last_slot_idx = last_valid_idx - 1
            timeline_slot = prediction_source_indices[last_slot_idx]
            
            pred_cat_scores = cat_logits[b, timeline_slot]
            true_cat_score = pred_cat_scores[true_cat_id].item()

            if args.mode == "poi":
                pred_spatial_scores = logits[b, timeline_slot] 
            else:
                loc_vector = predicted_loc_emb[b, last_slot_idx]
                pred_spatial_scores = torch.matmul(global_poi_spatial_embeddings, loc_vector) 
            
            true_spatial_score = pred_spatial_scores[true_target_poi].item()
            greater_spatial = (pred_spatial_scores > true_spatial_score).sum().item()
            equal_spatial_mask = (pred_spatial_scores == true_spatial_score)
            
            if args.mode != "poi" and cat_logits_global is not None:
                # Map the category score of every candidate POI using the global mapping tensor
                candidate_cat_scores = pred_cat_scores[cat_logits_global]
                
                # Out of the spatial ties, count how many have a strictly higher category score
                greater_cat_in_ties = (candidate_cat_scores[equal_spatial_mask] > true_cat_score).sum().item()
                
                # Count how many have an identical category score (includes the true target itself)
                equal_cat_in_ties = (candidate_cat_scores[equal_spatial_mask] == true_cat_score).sum().item()
                
                # Final pessimistic rank calculation resolving the structural ties
                rank = greater_spatial + greater_cat_in_ties + equal_cat_in_ties
            else:
                rank = greater_spatial + equal_spatial_mask.sum().item()

            test_mrr_sum += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    test_recall_hits[k] += 1
                        
            test_total_trajectories += 1
            
    return test_recall_hits, test_mrr_sum, test_total_trajectories