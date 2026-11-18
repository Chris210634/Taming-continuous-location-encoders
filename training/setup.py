# utils/setup.py
import torch
from torch.utils.data import DataLoader
from utils.latlon_dataset import LatLonTrajectoryDataset, LatLonCoordinateCollate, get_latlon_train_test_splits, prepare_user_trajectories_and_split


def _fetch_osm_regions(lat_min, lat_max, lon_min, lon_max, admin_level=10):
    """Fetch OSM admin + place polygons and return a deduplicated GeoDataFrame.
    Results are cached under stash/osm_regions/ keyed by bbox + admin_level."""
    import osmnx as ox
    import geopandas as gpd
    import pandas as pd
    import os

    cache_path = (f"stash/osm_regions/"
                  f"{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}_lvl{admin_level}.gpkg")
    if os.path.exists(cache_path):
        try:
            print(f"🗺  Loading cached OSM regions from {cache_path}")
            return gpd.read_file(cache_path, layer='regions')
        except Exception as e:
            print(f"  Cache read failed ({e}), re-fetching...")
            os.remove(cache_path)

    osm_bbox = (lon_min, lat_min, lon_max, lat_max)

    def _safe_fetch(tags, label):
        try:
            raw = ox.features_from_bbox(bbox=osm_bbox, tags=tags)
            polys = raw[raw.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
            print(f"  {label}: {len(polys)} polygons")
            return polys[['name', 'geometry']].copy() if len(polys) else gpd.GeoDataFrame(columns=['name', 'geometry'])
        except Exception as e:
            print(f"  {label} failed: {e}")
            return gpd.GeoDataFrame(columns=['name', 'geometry'])

    print("🗺  Fetching OSM regions for RegionLocationEncoder...")
    frames = []
    for lvl in range(7, admin_level + 1):
        frames.append(_safe_fetch({'boundary': 'administrative', 'admin_level': str(lvl)}, f'admin_level={lvl}'))
    frames.append(_safe_fetch({'place': ['neighbourhood', 'suburb', 'quarter', 'city_district']}, 'place=neighbourhood/suburb/quarter/city_district'))
    frames.append(_safe_fetch({'boundary': 'place'}, 'boundary=place'))

    regions = pd.concat(frames, ignore_index=True)
    regions = regions[regions['name'].notna()].drop_duplicates(subset='name')
    gdf = gpd.GeoDataFrame(regions, geometry='geometry', crs='EPSG:4326')
    print(f"  → Total unique regions: {len(gdf)}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf.to_file(cache_path, driver='GPKG', layer='regions')
    print(f"  → Cached to {cache_path}")
    return gdf

def build_location_encoder(args, device, unique_coords, num_unique, h3_info, train_loader=None):
    """
    Build and return a location encoder on `device`, with spatial_batch_norm warm-started.

    This is the single authoritative encoder-construction path shared by
    latlon_train.py and encoder_sanity.py.  warm_start_spatial_bn is called
    automatically for encoders that have a spatial_batch_norm layer, as long as
    a train_loader is provided and no pretrained weights are being loaded.
    """
    from models.location_encoder import (
        LocationEncoder, DiscreteLocationEncoder,
        H3LocationEncoder, RegionLocationEncoder, warm_start_spatial_bn,
    )
    baseline = args.baseline

    if baseline in ['rff', 'rff_sphere', 'rff_advanced', 'grid', 'hex', 'spherec', 'mlp']:
        raw_encoder = LocationEncoder(
            feature_type=baseline,
            out_dim=args.d_model,
            num_features=args.rff_features,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            num_bands=args.num_bands,
            hidden_dim=args.hidden_dim,
            device=str(device),
            num_layers_mlp=args.num_layers_mlp,
        )
        pretrained = getattr(args, 'pretrained_encoder_path', None)
        if pretrained:
            print(f"🔮 Loading pretrained encoder weights from: {pretrained}")
            checkpoint = torch.load(pretrained, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            encoder_state_dict = {
                k[len('encoder.'):]: v for k, v in state_dict.items() if k.startswith('encoder.')
            }
            raw_encoder.load_state_dict(encoder_state_dict, strict=True)
        if getattr(args, 'freeze_encoder', False):
            for param in raw_encoder.parameters():
                param.requires_grad = False
            print("❄️ Frozen encoder features.")
    elif baseline == 'discrete':
        raw_encoder = DiscreteLocationEncoder(
            num_unique=num_unique, d_model=args.d_model, unique_coords=unique_coords
        )
    elif baseline == 'h3':
        assert h3_info is not None, "h3_info missing — set --baseline h3"
        raw_encoder = H3LocationEncoder(
            num_unique_cells=h3_info['num_unique_cells'],
            d_model=args.d_model,
            h3_resolution=args.h3_resolution,
            cell_centroids=h3_info['cell_centroids'].to(device),
            cell_to_idx=h3_info['cell_to_idx'],
            unique_coord_h3_indices=h3_info['unique_coord_h3_indices'],
        )
    elif baseline == 'region':
        assert h3_info is not None and 'num_regions' in h3_info, "region_info missing — set --baseline region"
        raw_encoder = RegionLocationEncoder(
            num_regions=h3_info['num_regions'],
            d_model=args.d_model,
            region_centroids=h3_info['region_centroids'].to(device),
            poi_region_indices=h3_info['poi_region_indices'].to(device),
            poi_region_weights=h3_info['poi_region_weights'].to(device),
            unique_coords=unique_coords,
            regions_gdf=h3_info.get('regions_gdf'),
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    raw_encoder = raw_encoder.to(device)

    if (hasattr(raw_encoder, 'spatial_batch_norm')
            and train_loader is not None
            and not getattr(args, 'pretrained_encoder_path', None)):
        warm_start_spatial_bn(
            raw_encoder=raw_encoder,
            dataloader=train_loader,
            device=device,
            pad_val_coords=getattr(args, 'pad_val_coords', None),
        )

    return raw_encoder


def print_unseen_test_coordinates(train_dataset, test_dataset):
    """Computes and prints the count of unique test locations missing from the training set."""
    train_coords = {tuple(c) for sample in train_dataset.samples for c in sample['coords'].tolist()}
    test_coords = {tuple(c) for sample in test_dataset.samples for c in sample['coords'].tolist()}
    print(f"⚠️ Unique Test locations NOT seen during Training: {len(test_coords - train_coords)} (out of {len(test_coords)} total test locations)")

def initialize_pipeline_components(args):
    """Initializes device environment, datasets, loaders, padding protections, and deep models."""
    # Determine the execution target device with multi-GPU environment validation guards
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and 0 <= args.gpu_id < num_gpus:
        device_str = f"cuda:{args.gpu_id}"
        print(f"🔥 Multi-GPU cluster detected. Binding execution context to target physical device allocation: {args.gpu_id}")
    else:
        device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
        
    device = torch.device(device_str)
    print(f"📌 Pipeline processing runtime targeted allocation set to: {device}")

    csv_path = getattr(args, 'csv_path', None) or f"./data/{args.city}/{args.city}_checkins.csv"
    # train_ids, test_ids = get_latlon_train_test_splits(csv_path)

    csv_path, train_ids, test_ids = prepare_user_trajectories_and_split(
        csv_path=csv_path, seed=getattr(args, 'seed', 42))

    # H3-stratified POI split: compute test_poi_mask before building datasets
    test_poi_mask = None
    if getattr(args, 'h3_poi_split', False):
        from utils.latlon_dataset import compute_h3_poi_split, CSV_COLUMNS
        import pandas as pd
        df_all = pd.read_csv(csv_path, names=CSV_COLUMNS, header=0).dropna(subset=['latitude', 'longitude'])
        uc_all = torch.tensor(
            df_all[['latitude', 'longitude']].drop_duplicates().values.astype('float32'))
        test_poi_mask = compute_h3_poi_split(
            uc_all,
            h3_resolution=getattr(args, 'h3_split_resolution', 9),
            test_ratio=getattr(args, 'h3_split_test_ratio', 0.20),
            seed=getattr(args, 'seed', 42),
        )

    h3_resolution = getattr(args, 'h3_resolution', None) if getattr(args, 'baseline', None) == 'h3' else None
    train_dataset = LatLonTrajectoryDataset(csv_path, train_ids, is_train=True, mask_prob=args.mask_prob,
                                            h3_resolution=h3_resolution, test_poi_mask=test_poi_mask)
    bounds_dict = {
        'min_lat': train_dataset.min_lat, 'max_lat': train_dataset.max_lat,
        'min_lon': train_dataset.min_lon, 'max_lon': train_dataset.max_lon
    }

    # OSM region fetch — rebuild train_dataset so region spatial join runs on full vocab
    region_precomputed = None
    if getattr(args, 'baseline', None) == 'region':
        regions_gdf = _fetch_osm_regions(
            lat_min=train_dataset.min_lat, lat_max=train_dataset.max_lat,
            lon_min=train_dataset.min_lon, lon_max=train_dataset.max_lon,
            admin_level=getattr(args, 'region_admin_level', 10),
        )
        train_dataset = LatLonTrajectoryDataset(
            csv_path, train_ids, is_train=True, mask_prob=args.mask_prob,
            h3_resolution=h3_resolution,
            regions_gdf=regions_gdf,
            bounds=bounds_dict,
            test_poi_mask=test_poi_mask,
        )
        region_precomputed = {
            'poi_region_indices': train_dataset.poi_region_indices,
            'poi_region_weights': train_dataset.poi_region_weights,
            'region_centroids':   train_dataset.region_centroids,
            'num_regions':        train_dataset.num_regions,
            'max_R':              train_dataset.regions_max_R,
            'regions_gdf':        regions_gdf,
        }

    test_dataset = LatLonTrajectoryDataset(
        csv_path, test_ids, is_train=False,
        unique_coords_pool=train_dataset.unique_lat_lons,
        bounds=bounds_dict,
        h3_resolution=h3_resolution,
        h3_cell_to_idx=train_dataset.h3_cell_to_idx,
        region_precomputed=region_precomputed,
    )
    print_unseen_test_coordinates(train_dataset, test_dataset)
    
    collator = LatLonCoordinateCollate(
        padding_value_coords=args.pad_val_coords, 
        padding_value_targets=args.pad_val_targets
    )

    from torch.utils.data import ConcatDataset
    concat_train_dataset = ConcatDataset([train_dataset] * 10)
    train_loader = DataLoader(concat_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    unique_coords = train_dataset.unique_lat_lons.to(device)
    num_unique = unique_coords.size(0)
    
    center_lat = (train_dataset.min_lat + train_dataset.max_lat) / 2.0
    center_lon = (train_dataset.min_lon + train_dataset.max_lon) / 2.0
    neutral_pad = torch.tensor([center_lat, center_lon], device=device, dtype=torch.float32)
    
    h3_info = None
    if h3_resolution is not None:
        h3_info = {
            'cell_to_idx': train_dataset.h3_cell_to_idx,
            'cell_centroids': train_dataset.h3_cell_centroids,
            'unique_coord_h3_indices': train_dataset.unique_coord_h3_indices,
            'num_unique_cells': train_dataset.num_unique_h3_cells,
        }
    if region_precomputed is not None:
        if h3_info is None:
            h3_info = {}
        h3_info.update(region_precomputed)
        h3_info['train_dataset'] = train_dataset  # needed for poi_region_* buffers in encoder

    return device, train_loader, test_loader, unique_coords, num_unique, neutral_pad, h3_info