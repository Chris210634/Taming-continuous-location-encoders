"""Focused strict-unseen RFF rerun; legacy training paths are unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from importlib import metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from training.config import get_unified_args
from training.setup import build_location_encoder
from models.latlon_transformer import LatLonCausalTransformer
from utils.latlon_dataset import LatLonCoordinateCollate
from utils.evaluation import _heuristic_chunk
from utils.strict_unseen import make_split, evaluate, validate_split_options
from location_generalization.config import CORE_SOURCE_FILES
from location_generalization.selection import SELECTION_CRITERIA, best_validation_epoch, selection_score


SOURCE_MANIFEST_VERSION = 2
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root=ROOT):
    return {name: file_sha256(Path(root) / name) for name in CORE_SOURCE_FILES}


def verify_source_hashes(config, root=ROOT):
    """Fail closed when a historical config cannot establish complete provenance."""
    if config.get('source_manifest_version') != SOURCE_MANIFEST_VERSION:
        raise ValueError('This evaluation route requires a version-2 complete source manifest. '
                         'Historical runs must use their documented historical evaluation recipe; '
                         'their partial source hashes cannot establish exact reproduction here.')
    recorded = config.get('source_sha256', {})
    missing = set(CORE_SOURCE_FILES) - set(recorded)
    if missing:
        raise ValueError(f'Source manifest omits required dependencies: {sorted(missing)}')
    root = Path(root).resolve()
    for name, expected in recorded.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Source dependency is missing or outside the repository: {name}')
        if file_sha256(path) != expected:
            raise ValueError(f'Source dependency differs from training: {name}')
    return dict(recorded)


def runtime_versions():
    versions = {'python': sys.version.split()[0]}
    for package in ('torch', 'numpy', 'pandas', 'transformers', 'h3', 'scikit-learn', 'matplotlib'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def split_options(opts):
    return dict(train_fraction=opts.train_user_fraction,
                validation_fraction=opts.validation_user_fraction,
                max_length=opts.max_sequence_length,
                heldout_validation_fraction=opts.heldout_validation_fraction)


def build_model(args, opts, train_vocabulary, base_loader, device):
    """Construct the same architecture for training and strict checkpoint loading."""
    train_coords = train_vocabulary.to(device)
    if args.baseline == 'h3':
        from utils.strict_encoders import build_strict_h3
        encoder = build_strict_h3(train_coords, args.d_model, args.h3_resolution, device)
    else:
        encoder = build_location_encoder(args, device, train_coords, len(train_coords), None, base_loader)
    if opts.learn_band_scales:
        if args.baseline != 'rff':
            raise ValueError('Matched band-scale adaptation requires paired RFF baseline')
        from utils.strict_encoders import BoundedBandFourierFeatures
        encoder.encoder = BoundedBandFourierFeatures(encoder.encoder, args.num_bands, opts.band_scale_limit)
    model = LatLonCausalTransformer(encoder, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        learnable_alpha=args.heuristic_loss, backbone_type=args.backbone_type,
        max_len=max(2048, opts.max_sequence_length - 1)).to(device)
    return encoder, model


def training_counts(dataset, vocabulary, repeats, popularity):
    counts = torch.zeros(len(vocabulary))
    for sample in dataset.samples:
        counts.scatter_add_(0, sample['target_ids'], torch.ones(len(sample['target_ids'])))
    # Retain the existing loader-count convention for the independent prior.
    counts *= repeats
    if not popularity:
        counts.zero_()
    return counts


def evaluate_checkpoint(source_run, output_dir, csv_path=None, gpu_id=None):
    """Evaluate a frozen completed run once into a new directory, without fitting."""
    source, output = Path(source_run), Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Evaluation requires a fresh output directory: {output}')
    config = json.loads((source / 'config.json').read_text())
    verified_sources = verify_source_hashes(config)
    completed = json.loads((source / 'training_complete.json').read_text())
    for name in ('config.json', 'split.json', 'vocabularies.pt', 'best.pt', 'validation.jsonl'):
        if completed.get('artifact_sha256', {}).get(name) != file_sha256(source / name):
            raise ValueError(f'Training artifact differs from completed run: {name}')
    args = argparse.Namespace(**config['training'])
    opts = argparse.Namespace(**config['strict'])
    validate_split_options(**split_options(opts))
    saved = torch.load(source / 'best.pt', map_location='cpu', weights_only=True)
    history = [json.loads(line) for line in (source / 'validation.jsonl').read_text().splitlines() if line.strip()]
    validate_checkpoint_selection(saved, history, opts.selection_metric, completed)
    if config['selection']['metric'] != opts.selection_metric:
        raise ValueError('Saved selection configuration disagrees with the training criterion.')
    original_manifest = json.loads((source / 'split.json').read_text())
    raw_path = Path(csv_path or config['data']['csv_path'])
    if file_sha256(raw_path) != original_manifest['raw_sha256']:
        raise ValueError('Raw CSV differs from the dataset used for training.')
    if gpu_id is not None:
        args.gpu_id = gpu_id
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    datasets, vocabs, manifest = make_split(raw_path, output, seed=opts.split_seed,
        resolution=args.h3_split_resolution, fraction=args.h3_split_test_ratio, **split_options(opts))
    if json.loads(json.dumps(manifest)) != original_manifest:
        raise ValueError('Recreated split differs from the saved training split.')
    original_vocabs = torch.load(source / 'vocabularies.pt', map_location='cpu', weights_only=True)
    if not all(torch.equal(vocabulary, original_vocabs[name])
               for name, vocabulary in zip(('train', 'validation', 'test'), vocabs)):
        raise ValueError('Recreated candidate vocabularies differ from training.')
    encoder, model = build_model(args, opts, vocabs[0], None, device)
    model.load_state_dict(saved['model'], strict=True)
    expected_counts = training_counts(datasets[0], vocabs[0], opts.train_repeats, args.heuristic_popularity)
    if not torch.equal(saved['train_counts'].cpu(), expected_counts):
        raise ValueError('Checkpoint popularity counts differ from the reconstructed training data.')
    collate = LatLonCoordinateCollate(args.pad_val_coords, args.pad_val_targets)
    loader = DataLoader(datasets[2], batch_size=opts.eval_batch_size, collate_fn=collate)
    familiar = None
    if args.baseline == 'h3':
        from utils.strict_encoders import familiar_h3_cells
        familiar = familiar_h3_cells(vocabs[2], encoder)
    metrics = evaluate(model, loader, vocabs[2], vocabs[0], saved['train_counts'], device, args,
        reserved_locations=manifest['test_locations'], ablations=True,
        per_user_path=output / 'per_user.json', familiar_cell_mask=familiar)
    report = dict(selected_epoch=saved['epoch'], selection_metric=saved['selection_metric'],
        validation_selection_score=saved['selection_score'], tie_policy=args.eval_tie_policy,
        metrics=metrics, source_run=str(source.resolve()), training_performed=False,
        checkpoint_sha256=file_sha256(source / 'best.pt'), raw_sha256=manifest['raw_sha256'],
        source_manifest_version=SOURCE_MANIFEST_VERSION, source_sha256=verified_sources,
        source_verification='all recorded core dependencies match training',
        runtime=runtime_versions(), training_runtime=config.get('runtime'), device=str(device))
    (output / 'test.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FINAL_TEST', json.dumps(metrics), flush=True)
    return report


def validate_checkpoint_selection(saved, history, criterion, completed=None):
    """Verify the checkpoint is the earliest best validation epoch for its metric."""
    best = best_validation_epoch(history, criterion)
    expected = dict(selected_epoch=best['epoch'], selection_metric=criterion,
                    validation_selection_score=best['selection_score'])
    actual = dict(selected_epoch=saved.get('epoch'), selection_metric=saved.get('selection_metric'),
                  validation_selection_score=saved.get('selection_score'))
    if actual != expected:
        raise ValueError('Checkpoint is not the earliest maximum of the saved validation criterion.')
    if completed is not None and any(completed.get(key) != value for key, value in expected.items()):
        raise ValueError('Completed-run selection metadata disagrees with the checkpoint and validation history.')
    return expected


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--output_dir')
    extra.add_argument('--selection_metric', choices=list(SELECTION_CRITERIA), default='overall_model_r10')
    extra.add_argument('--split_seed', type=int, default=42)
    extra.add_argument('--eval_batch_size', type=int, default=16)
    extra.add_argument('--train_repeats', type=int, default=10)
    extra.add_argument('--prepare_only', action='store_true')
    extra.add_argument('--eval_tie_policy', choices=['worst','expected_random'], default='worst')
    extra.add_argument('--learn_band_scales', action='store_true')
    extra.add_argument('--band_scale_limit', type=float, default=2.)
    extra.add_argument('--no_heuristic_popularity', action='store_true')
    extra.add_argument('--train_user_fraction', type=float, default=.8)
    extra.add_argument('--validation_user_fraction', type=float, default=.1)
    extra.add_argument('--max_sequence_length', type=int, default=128)
    extra.add_argument('--heldout_validation_fraction', type=float, default=.5)
    extra.add_argument('--skip_test', action='store_true', help='Save the selected checkpoint without evaluating test data.')
    extra.add_argument('--evaluate_only', metavar='SOURCE_RUN', help='Evaluate a completed frozen run without training.')
    extra.add_argument('--evaluation_output_dir')
    extra.add_argument('--evaluation_csv_path')
    extra.add_argument('--evaluation_gpu_id', type=int)
    opts, remaining = extra.parse_known_args()
    if opts.evaluate_only:
        allowed = {'--evaluate_only', '--evaluation_output_dir', '--evaluation_csv_path', '--evaluation_gpu_id'}
        unsupported = [token.split('=', 1)[0] for token in sys.argv[1:]
                       if token.startswith('--') and token.split('=', 1)[0] not in allowed]
        if remaining or unsupported:
            extra.error('Evaluation restores saved settings; only --evaluation_* overrides are accepted.')
        if not opts.evaluation_output_dir:
            extra.error('--evaluate_only requires --evaluation_output_dir.')
        return evaluate_checkpoint(opts.evaluate_only, opts.evaluation_output_dir,
                                   opts.evaluation_csv_path, opts.evaluation_gpu_id)
    if not opts.output_dir:
        extra.error('Training requires --output_dir.')
    if any(value is not None for value in (opts.evaluation_output_dir, opts.evaluation_csv_path, opts.evaluation_gpu_id)):
        extra.error('--evaluation_* arguments require --evaluate_only.')
    validate_split_options(**split_options(opts))
    sys.argv = [sys.argv[0]] + remaining
    args = get_unified_args()
    args.eval_tie_policy = opts.eval_tie_policy
    if opts.no_heuristic_popularity: args.heuristic_popularity = False
    if args.baseline not in ['rff','rff_sphere','rff_advanced','mlp','grid','hex','spherec','h3']:
        raise ValueError('This focused runner supports continuous encoders and the training-only H3 baseline.')
    if args.multistep_k != 1 or args.use_time_fusion or args.task != 'next_event' or args.mask_prob or args.num_neg_samples:
        raise ValueError('Strict v1 uses one-step, full-softmax, unmasked coordinate training.')
    if args.pretrained_encoder_path:
        raise ValueError('Pretrained weights could already expose held-out coordinates.')
    output = Path(opts.output_dir)
    if (output/'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    raw_path = Path(args.csv_path or ROOT/f'data/{args.city}/{args.city}_checkins.csv').resolve()
    datasets, vocabs, manifest = make_split(raw_path,
        output, seed=opts.split_seed, resolution=args.h3_split_resolution, fraction=args.h3_split_test_ratio,
        **split_options(opts))
    if opts.selection_metric == 'first_visit_unseen_trained_r100' and not args.heuristic_loss:
        raise ValueError('Learned-weight selection requires --heuristic_loss.')
    selection_weight = ('learned from each sequence' if opts.selection_metric == 'first_visit_unseen_trained_r100'
                        else None if opts.selection_metric in ('overall_model_r10', 'first_visit_unseen_model_r100')
                        else args.heuristic_alpha)
    config = dict(training=vars(args), strict=vars(opts), selection=dict(metric=opts.selection_metric, split='validation only', alpha=selection_weight, exact_ties='retain earliest epoch', early_stopping='same selection metric'),
        source_manifest_version=SOURCE_MANIFEST_VERSION, source_sha256=source_hashes(),
        data=dict(csv_path=str(raw_path), raw_sha256=manifest['raw_sha256']), runtime=runtime_versions())
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('SPLIT', json.dumps({k:manifest[k] for k in ['user_counts','sequence_counts','candidate_counts','heldout_validation_count','heldout_test_count']}),flush=True)
    if opts.prepare_only: return
    if args.epochs < 1 or opts.train_repeats < 1: raise ValueError('Positive training budget required.')
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id>=0 else 'cpu')
    collate=LatLonCoordinateCollate(args.pad_val_coords,args.pad_val_targets)
    base_loader=DataLoader(datasets[0],batch_size=args.batch_size,collate_fn=collate)
    train_loader=DataLoader(ConcatDataset([datasets[0]]*opts.train_repeats),batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(datasets[1],batch_size=opts.eval_batch_size,collate_fn=collate)
    train_coords=vocabs[0].to(device)
    encoder, model = build_model(args, opts, vocabs[0], base_loader, device)
    if args.baseline == 'h3':
        from utils.strict_encoders import familiar_h3_cells
        config['h3_protocol'] = dict(training_cells=len(encoder.cell_to_idx),
            unseen_cell_fallback='nearest training-cell geographic centroid in angular coordinates')
        (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    encoder_params=list(encoder.parameters()); head_params=list(model.alpha_head.parameters()) if args.heuristic_loss else []
    special={id(p) for p in encoder_params+head_params}
    parameter_groups=[dict(params=encoder_params,mult=args.encoder_lr_mult),
        dict(params=[p for p in model.parameters() if id(p) not in special],mult=1.),
        dict(params=head_params,mult=args.head_lr_mult)]
    optimizer=torch.optim.AdamW(parameter_groups,lr=args.lr,weight_decay=args.weight_decay)
    counts = training_counts(datasets[0], vocabs[0], opts.train_repeats, args.heuristic_popularity)
    pop=torch.log1p(counts).to(device)
    best=-1.; stale=0
    for epoch in range(1,args.epochs+1):
        model.train()
        if hasattr(encoder, 'spatial_batch_norm'): encoder.spatial_batch_norm.eval()
        schedule=1. if epoch<=args.lr_step1_epochs else args.lr_step2_mult if epoch<=args.lr_step2_epochs else args.lr_step3_mult
        for group in optimizer.param_groups: group['lr']=args.lr*schedule*group['mult']
        losses=[]
        for batch in train_loader:
            inputs=batch['coords'][:,:-1].to(device).clone(); src=batch['targets'][:,:-1].to(device)
            targets=batch['targets'][:,1:].to(device); lengths=batch['lengths'].to(device)-1
            inputs[src<0]=train_coords.mean(0)
            optimizer.zero_grad(set_to_none=True)
            input_kw = dict(h3_indices=encoder.unique_coord_h3_indices[src.clamp_min(0)]) if args.baseline == 'h3' else {}
            embedded=encoder(inputs,**input_kw).masked_fill((src<0).unsqueeze(-1),0.)
            hidden=model(lengths=lengths,embeddings=embedded)
            # The only candidate tensor encoded in a training forward pass is train_coords.
            candidate_kw = dict(h3_indices=encoder.unique_coord_h3_indices) if args.baseline == 'h3' else {}
            candidates=F.normalize(encoder(train_coords,**candidate_kw),dim=-1)
            logits=F.normalize(hidden,dim=-1) @ candidates.T / args.temperature
            if args.heuristic_loss:
                with torch.no_grad():
                    prior=_heuristic_chunk(inputs,src,(src>=0).float(),train_coords,pop,args.heuristic_gamma,args.heuristic_beta,0)
                logits=logits+torch.sigmoid(model.alpha_head(hidden))*prior
            loss=F.cross_entropy(logits.reshape(-1,len(train_coords)),targets.reshape(-1),ignore_index=args.pad_val_targets)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; aborting run.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm);optimizer.step()
            losses.append(loss.item())
        # Free full-softmax tensors before validation on the larger candidate pool.
        del logits, candidates, hidden, embedded, loss
        if args.heuristic_loss: del prior
        if device.type=='cuda': torch.cuda.empty_cache()
        metrics=evaluate(model,val_loader,vocabs[1],vocabs[0],counts,device,args)
        score=selection_score(metrics,opts.selection_metric)
        with (output/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),selection_metric=opts.selection_metric,selection_score=score,metrics=metrics))+'\n')
        if opts.learn_band_scales:
            with (output/'band_scales.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch,multipliers=encoder.encoder.multipliers().detach().cpu().tolist()))+'\n')
        print(f'Epoch {epoch}: loss={np.mean(losses):.5f}; validation {opts.selection_metric}={score:.5f}',flush=True)
        if score>best:
            best=score;stale=0
            torch.save(dict(model=model.state_dict(),epoch=epoch,train_counts=counts,selection_metric=opts.selection_metric,selection_score=score),output/'best.pt')
        else: stale+=1
        if stale>=args.early_stopping_patience:break
    # Complete the immutable training record before any optional test evaluation.
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    history = [json.loads(line) for line in (output / 'validation.jsonl').read_text().splitlines() if line.strip()]
    selected = validate_checkpoint_selection(saved, history, opts.selection_metric)
    completed = dict(**selected, test_evaluation_requested=not opts.skip_test,
        artifact_sha256={name: file_sha256(output / name) for name in
                         ('config.json', 'split.json', 'vocabularies.pt', 'best.pt', 'validation.jsonl')})
    (output / 'training_complete.json').write_text(json.dumps(completed, indent=2) + '\n')
    if opts.skip_test:
        print('TRAINING_COMPLETE', str(output), 'test evaluation skipped', flush=True)
        return
    # No test scores are computed before the checkpoint and alpha are fixed.
    model.load_state_dict(saved['model'], strict=True)
    test_loader=DataLoader(datasets[2],batch_size=opts.eval_batch_size,collate_fn=collate)
    metrics=evaluate(model,test_loader,vocabs[2],vocabs[0],counts,device,args,
        reserved_locations=manifest['test_locations'],ablations=True,per_user_path=output/'per_user.json',
        familiar_cell_mask=familiar_h3_cells(vocabs[2],encoder) if args.baseline == 'h3' else None)
    (output/'test.json').write_text(json.dumps(dict(selected_epoch=saved['epoch'],selection_metric=opts.selection_metric,validation_selection_score=saved['selection_score'],tie_policy=args.eval_tie_policy,metrics=metrics),indent=2)+'\n')
    print('FINAL_TEST',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
