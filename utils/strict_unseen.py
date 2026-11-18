"""Strict geographic holdouts, separate from the legacy label-masking protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def coordinate_keys(frame):
    # Match the precision actually presented to the encoder before deduplication.
    return [tuple(map(float, row)) for row in frame[['latitude', 'longitude']].to_numpy(dtype=np.float32)]


def split_segments(frame, excluded, max_length=128):
    """Drop excluded events and break context at each gap; never stitch across it."""
    segments = []
    for _, group in frame.groupby(['user_id', 'trail_id'], sort=False):
        current = []
        for idx, key in zip(group.index, coordinate_keys(group)):
            if key in excluded:
                if len(current) >= 2:
                    segments.append(frame.loc[current])
                current = []
            else:
                current.append(idx)
                if len(current) == max_length:
                    segments.append(frame.loc[current])
                    current = []
        if len(current) >= 2:
            segments.append(frame.loc[current])
    return segments


class StrictTrajectoryDataset(Dataset):
    def __init__(self, segments, vocabulary):
        index = {tuple(c): i for i, c in enumerate(vocabulary.tolist())}
        self.samples = []
        for segment in segments:
            coords = torch.tensor(np.asarray(coordinate_keys(segment)), dtype=torch.float32)
            self.samples.append(dict(coords=coords, user_id=str(segment.user_id.iloc[0]),
                target_ids=torch.tensor([index[tuple(c)] for c in coords.tolist()]),
                timestamps=torch.tensor(segment.timestamp.astype('int64').to_numpy() // 10**9)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def validate_split_options(train_fraction=.8, validation_fraction=.1,
                           max_length=128, heldout_validation_fraction=.5):
    """Validate requested split sizes before reading data or creating artifacts."""
    for name, value in [('train_fraction', train_fraction),
                        ('validation_fraction', validation_fraction),
                        ('heldout_validation_fraction', heldout_validation_fraction)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'{name} must be finite and strictly between zero and one.')
    if train_fraction + validation_fraction >= 1:
        raise ValueError('train_fraction + validation_fraction must leave a positive test fraction.')
    if isinstance(max_length, (bool, np.bool_)) or not isinstance(max_length, (int, np.integer)) or max_length < 2:
        raise ValueError('max_length must be an integer of at least two events.')


def make_split(csv_path, output_dir, seed=42, resolution=9, fraction=.2, *,
               train_fraction=.8, validation_fraction=.1, max_length=128,
               heldout_validation_fraction=.5):
    """Disjoint users and reserved coordinate holdouts within H3 cells.

    Only the split constructor may inspect all coordinates. Training consumes its
    own vocabulary. Validation never receives reserved test coordinates, even as
    negatives; final test receives the full catalog after model selection.
    Defaults preserve the original 80/10/10 split and its exact manifest shape.
    Configured alternatives record all four options in ``split_options``.
    """
    import h3
    validate_split_options(train_fraction, validation_fraction, max_length, heldout_validation_fraction)
    if not 0 < fraction < 1:
        raise ValueError('Holdout fraction must be between zero and one.')
    frame = pd.read_csv(csv_path).dropna(subset=['latitude', 'longitude']).copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True).astype('datetime64[ns, UTC]')
    frame = frame.sort_values(['user_id', 'timestamp'], kind='stable').reset_index(drop=True)
    all_keys = sorted(set(coordinate_keys(frame)))
    cells = {}
    for key in all_keys:
        cells.setdefault(h3.latlng_to_cell(*map(float, key), resolution), []).append(key)
    rng = np.random.default_rng(seed)
    heldout = []
    for cell in sorted(cells):
        keys = cells[cell]
        if len(keys) >= 2:
            count = min(len(keys)-1, int(np.ceil(fraction*len(keys))))
            heldout.extend(keys[i] for i in rng.permutation(len(keys))[:count])
    if len(heldout) < 2:
        raise ValueError('Not enough held-out coordinates for separate validation/test sets.')
    heldout = [heldout[i] for i in rng.permutation(len(heldout))]
    n_val_locations = int(heldout_validation_fraction * len(heldout))
    if n_val_locations < 1 or n_val_locations >= len(heldout):
        raise ValueError('Requested holdout allocation leaves validation or test without reserved coordinates.')
    val_locations = set(heldout[:n_val_locations])
    test_locations = set(heldout[n_val_locations:])
    users = np.asarray(sorted(frame.user_id.unique()))
    shuffled = np.random.default_rng(seed).permutation(users)
    n_train, n_val = int(train_fraction*len(users)), int(validation_fraction*len(users))
    if n_train < 1 or n_val < 1 or n_train+n_val >= len(users):
        raise ValueError('Requested user fractions must allocate at least one user to each split.')
    user_sets = [set(shuffled[:n_train]), set(shuffled[n_train:n_train+n_val]), set(shuffled[n_train+n_val:])]
    segments = [split_segments(frame[frame.user_id.isin(us)], ex, max_length=max_length) for us, ex in
                zip(user_sets, [val_locations | test_locations, test_locations, set()])]
    if not all(segments):
        raise ValueError('A split has no two-event sequences after exclusion.')
    train_keys = sorted({key for seg in segments[0] for key in coordinate_keys(seg)})
    val_keys = train_keys + sorted(({key for seg in segments[1] for key in coordinate_keys(seg)} | val_locations) - set(train_keys))
    test_keys = train_keys + sorted(set(all_keys)-set(train_keys))
    assert not set(train_keys) & set(heldout)
    assert not set(val_keys) & test_locations
    vocabularies = [torch.tensor(k, dtype=torch.float32) for k in [train_keys, val_keys, test_keys]]
    datasets = [StrictTrajectoryDataset(seg, vocab) for seg, vocab in zip(segments, vocabularies)]
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol='strict-unseen-v1', split_seed=seed, h3_resolution=resolution,
        requested_holdout_fraction=fraction, raw_sha256=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        user_counts=list(map(len,user_sets)), sequence_counts=list(map(len,datasets)),
        candidate_counts=list(map(len,vocabularies)), heldout_validation_count=len(val_locations),
        heldout_test_count=len(test_locations), segmentation=f'original trails; break at excluded events and {max_length}-event limit',
        training_exposure='training-only inputs, targets, candidates, and normalization',
        users=[[str(u) for u in sorted(us)] for us in user_sets],
        validation_locations=sorted(val_locations), test_locations=sorted(test_locations))
    if (train_fraction, validation_fraction, max_length, heldout_validation_fraction) != (.8, .1, 128, .5):
        manifest['split_options'] = dict(train_fraction=float(train_fraction),
            validation_fraction=float(validation_fraction), max_length=int(max_length),
            heldout_validation_fraction=float(heldout_validation_fraction))
    (output/'split.json').write_text(json.dumps(manifest, indent=2)+'\n')
    torch.save(dict(train=vocabularies[0], validation=vocabularies[1], test=vocabularies[2]), output/'vocabularies.pt')
    return datasets, vocabularies, manifest


def expected_random_tie_metrics(strictly_above, tie_size, k_values, harmonic=None):
    """Exact expected metrics for uniformly permuting candidates within a tie.

    a candidates precede the tie; the target is uniform over ranks a+1..a+m.
    Counts include the target in m. Never use reciprocal expected rank for MRR.
    """
    a = strictly_above.long()
    m = tie_size.long()
    if torch.any(a < 0) or torch.any(m < 1):
        raise ValueError('Expected nonnegative preceding counts and nonempty ties')
    if harmonic is None:
        n = int((a + m).max().item()) if a.numel() else 0
        harmonic = torch.cat([torch.zeros(1, device=a.device, dtype=torch.float64),
            torch.arange(1, n + 1, device=a.device, dtype=torch.float64).reciprocal().cumsum(0)])
    rr = (harmonic[a + m] - harmonic[a]) / m
    hits = {str(k): ((k - a).double() / m).clamp(0., 1.) for k in k_values}
    return rr, hits


def alpha_stream_name(baseline, alpha):
    return f'alpha_{baseline}_{alpha:.6f}'


def alpha_score_streams(model_scores, baselines, alphas):
    """Stream one blend at a time; raw endpoints preserve exact baseline ties."""
    def z(x):
        return (x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,keepdim=True,unbiased=False)+1e-8)
    zm = z(model_scores)
    for name, baseline in baselines.items():
        zb = z(baseline)
        for alpha in alphas:
            scores = model_scores if alpha == 0 else baseline if alpha == 1 else (1-alpha)*zm+alpha*zb
            yield alpha_stream_name(name,alpha), scores


def trained_combination_scores(model_scores, prior_scores, hidden, alpha_head):
    """Use the same sequence-dependent combination optimized during training."""
    sequence_weight = torch.sigmoid(alpha_head(hidden))
    return model_scores + sequence_weight * prior_scores, sequence_weight


@torch.no_grad()
def evaluate(model, loader, vocabulary, train_vocabulary, train_counts, device, args, reserved_locations=None, ablations=False, candidate_training_indices=None, per_user_path=None, familiar_cell_mask=None):
    """Same candidates, worst-rank tie convention, and FV/RV groups for all scores."""
    from utils.evaluation import _heuristic_chunk
    model.eval()
    coords = vocabulary.to(device)
    ntrain = len(train_vocabulary)
    if candidate_training_indices is None:
        assert torch.equal(vocabulary[:ntrain], train_vocabulary)
        training_index = torch.full((len(coords),), -1, device=device, dtype=torch.long)
        training_index[:ntrain] = torch.arange(ntrain, device=device)
    else:
        training_index = torch.as_tensor(candidate_training_indices, device=device, dtype=torch.long)
        expected = {tuple(c): i for i,c in enumerate(train_vocabulary.tolist())}
        assert training_index.tolist() == [expected.get(tuple(c), -1) for c in vocabulary.tolist()]
    candidate_seen = training_index >= 0
    counts = torch.zeros(len(coords), device=device)
    counts[candidate_seen] = train_counts.to(device)[training_index[candidate_seen]]
    pop = torch.log1p(counts)
    embeds = torch.nn.functional.normalize(model.location_encoder(coords), dim=-1)
    tie_policy = getattr(args, 'eval_tie_policy', 'worst')
    if tie_policy not in ('worst', 'expected_random'):
        raise ValueError(f'Unknown tie policy: {tie_policy}')
    harmonic = torch.cat([torch.zeros(1, device=device, dtype=torch.float64),
        torch.arange(1, len(coords)+1, device=device, dtype=torch.float64).reciprocal().cumsum(0)]) if tie_policy == 'expected_random' else None
    groups = ['all', 'seen', 'unseen', 'first_visit_seen', 'repeat_visit_seen', 'first_visit_unseen', 'repeat_visit_unseen']
    reserved_mask = None
    if reserved_locations is not None:
        reserved = set(map(tuple, reserved_locations))
        reserved_mask = torch.tensor([tuple(c) in reserved for c in vocabulary.tolist()], device=device)
        groups += ['reserved_test', 'first_visit_reserved_test', 'repeat_visit_reserved_test']
    if familiar_cell_mask is not None:
        familiar_cell_mask = torch.as_tensor(familiar_cell_mask, device=device, dtype=torch.bool)
        assert len(familiar_cell_mask) == len(coords)
        groups += ['first_visit_unseen_familiar_cell', 'first_visit_unseen_new_cell']
    alphas = list(getattr(args, 'alpha_grid', []))
    if any(not np.isfinite(a) or a < 0 or a > 1 for a in alphas):
        raise ValueError('Alpha values must be finite and in [0,1]')
    if len({alpha_stream_name('spatial',a) for a in alphas}) != len(alphas):
        raise ValueError('Alpha grid contains duplicate stream names')
    user_totals = {}
    sample_offset = 0
    if per_user_path is not None:
        from torch.utils.data import SequentialSampler
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError('Per-user evaluation requires an ordered, non-shuffled loader')
    use_trained_combination = hasattr(model, 'alpha_head')
    streams = ['model', 'spatial', 'prior', 'ensemble']
    if use_trained_combination:
        streams += ['trained_combination']
    if ablations:
        streams += ['distance_history', 'model_spatial', 'model_distance_history']
    streams += [alpha_stream_name(b,a) for b in ['spatial','prior'] for a in alphas]
    results = {name: {g: dict(n=0, reciprocal_rank_sum=0., hits={str(k):0 for k in args.k_values}) for g in groups}
               for name in streams}
    for batch in loader:
        inputs = batch['coords'][:,:-1].to(device)
        src = batch['targets'][:,:-1].to(device)
        targets = batch['targets'][:,1:].to(device)
        lengths = batch['lengths'].to(device)-1
        valid = targets >= 0
        inputs = inputs.clone(); inputs[src<0] = train_vocabulary.mean(0).to(device)
        input_embeds = model.location_encoder(inputs).masked_fill((src<0).unsqueeze(-1), 0.)
        hidden = model(lengths=lengths, embeddings=input_embeds)
        sims = torch.nn.functional.normalize(hidden,dim=-1) @ embeds.T / args.temperature
        prior = _heuristic_chunk(inputs,src,(src>=0).float(),coords,pop,args.heuristic_gamma,args.heuristic_beta,0)
        spatial = -torch.linalg.vector_norm(inputs.unsqueeze(2)-coords,dim=-1)
        def z(x): return (x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,keepdim=True,unbiased=False)+1e-8)
        ensemble = (1-args.heuristic_alpha)*z(sims)+args.heuristic_alpha*z(prior)
        trained_combination = None
        if use_trained_combination:
            trained_combination, _ = trained_combination_scores(
                sims, prior, hidden, model.alpha_head)
        causal = torch.ones(src.shape[1],src.shape[1],device=device,dtype=torch.bool).tril()
        repeat = ((targets.unsqueeze(-1)==src.unsqueeze(1)) & causal).any(-1)
        unseen = ~candidate_seen[targets.clamp_min(0)]
        masks = dict(all=valid, seen=valid & ~unseen, unseen=valid & unseen,
                     first_visit_seen=valid & ~unseen & ~repeat, repeat_visit_seen=valid & ~unseen & repeat,
                     first_visit_unseen=valid & unseen & ~repeat, repeat_visit_unseen=valid & unseen & repeat)
        if reserved_mask is not None:
            held = valid & reserved_mask[targets.clamp_min(0)]
            masks.update(reserved_test=held, first_visit_reserved_test=held & ~repeat, repeat_visit_reserved_test=held & repeat)
        if familiar_cell_mask is not None:
            familiar = familiar_cell_mask[targets.clamp_min(0)]
            masks.update(first_visit_unseen_familiar_cell=valid & unseen & ~repeat & familiar,
                         first_visit_unseen_new_cell=valid & unseen & ~repeat & ~familiar)
        score_streams = [('model',sims),('spatial',spatial),('prior',prior),('ensemble',ensemble)]
        if trained_combination is not None:
            score_streams += [('trained_combination', trained_combination)]
        if ablations:
            no_pop = _heuristic_chunk(inputs,src,(src>=0).float(),coords,torch.zeros_like(pop),args.heuristic_gamma,args.heuristic_beta,0)
            score_streams += [('distance_history',no_pop),
                ('model_spatial',(1-args.heuristic_alpha)*z(sims)+args.heuristic_alpha*z(spatial)),
                ('model_distance_history',(1-args.heuristic_alpha)*z(sims)+args.heuristic_alpha*z(no_pop))]
        if alphas:
            from itertools import chain
            score_streams = chain(score_streams, alpha_score_streams(sims, {'spatial':spatial, 'prior':prior}, alphas))
        for name, scores in score_streams:
            truth = scores.gather(-1,targets.clamp_min(0).unsqueeze(-1))
            ranks = (scores >= truth).sum(-1)
            if tie_policy == 'expected_random':
                above = (scores > truth).sum(-1)
                rr, hits = expected_random_tie_metrics(above, ranks-above, args.k_values, harmonic)
            else:
                rr = ranks.float().reciprocal()
                hits = {str(k): (ranks <= k) for k in args.k_values}
            for group, mask in masks.items():
                out=results[name][group]; out['n']+=int(mask.sum().item())
                out['reciprocal_rank_sum']+=rr[mask].sum().item()
                for k in args.k_values: out['hits'][str(k)]+=hits[str(k)][mask].sum().item()
                if per_user_path is not None:
                    ns = mask.sum(-1).cpu().tolist()
                    rs = (rr * mask).sum(-1).cpu().tolist()
                    hs = {k:(value * mask).sum(-1).cpu().tolist() for k,value in hits.items()}
                    for row,n in enumerate(ns):
                        if not n: continue
                        uid = loader.dataset.samples[sample_offset+row]['user_id']
                        u = user_totals.setdefault(name, {}).setdefault(group, {}).setdefault(uid,
                            dict(n=0, reciprocal_rank_sum=0., hits={str(k):0. for k in args.k_values}))
                        u['n'] += n; u['reciprocal_rank_sum'] += rs[row]
                        for k in hs: u['hits'][k] += hs[k][row]
        sample_offset += len(targets)
    for scores in results.values():
        for out in scores.values():
            out['mrr']=out['reciprocal_rank_sum']/out['n'] if out['n'] else None
            out['recall']={k:h/out['n'] if out['n'] else None for k,h in out['hits'].items()}
    if per_user_path is not None:
        Path(per_user_path).write_text(json.dumps(user_totals)+'\n')
    return results
