import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree
import time
# from shapely.geometry import Point

CSV_COLUMNS = [
    "trail_id", "user_id", "venue_id", "latitude", "longitude", "name", 
    "address", "venue_category", "venue_category_id", "venue_category_id_code", 
    "venue_city", "venue_city_latitude", "venue_city_longitude", "venue_country", "timestamp"
]

def get_topk_nearest_pois(unique_coords, pois_gdf, k=10):
    unique_gdf = gpd.GeoDataFrame(
        pd.DataFrame(unique_coords.numpy(), columns=['lat', 'lon']),
        geometry=[Point(xy) for xy in zip(unique_coords[:,1], unique_coords[:,0])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    pois_proj = pois_gdf.to_crs(epsg=3857)
    
    unique_coords_array = np.array(list(zip(unique_gdf.geometry.x, unique_gdf.geometry.y)))
    poi_centroids = pois_proj.geometry.centroid
    poi_coords_array = np.array(list(zip(poi_centroids.x, poi_centroids.y)))
    
    tree = BallTree(poi_coords_array, leaf_size=30)
    dists, indices = tree.query(unique_coords_array, k=k)
    
    knn_results = pd.DataFrame({
        'unique_idx': np.repeat(np.arange(len(unique_gdf)), k),
        'poi_idx': indices.flatten(),
        'distance': dists.flatten()
    })
    
    # Merge amenity info
    knn_results = knn_results.join(pois_proj[['amenity']].reset_index(drop=True), on='poi_idx')
    return knn_results

def compute_h3_poi_split(unique_coords, h3_resolution=9, test_ratio=0.2, seed=42):
    """
    For each H3 cell, withholds test_ratio of its POIs from training supervision.

    Args:
        unique_coords: torch.Tensor (N, 2) — all unique (lat, lon) pairs
        h3_resolution: H3 hierarchy level (7 ≈ 5km, 9 ≈ 200m, 11 ≈ 25m)
        test_ratio:    fraction of POIs per cell to withhold (e.g. 0.2 = 20%)
        seed:          random seed for reproducibility

    Returns:
        test_poi_mask: torch.BoolTensor (N,) — True = held-out test POI
    """
    import h3 as h3lib
    rng = np.random.default_rng(seed)

    coords_np = unique_coords.numpy()
    N = len(coords_np)

    cell_to_poi_indices = {}
    for i, (lat, lon) in enumerate(coords_np):
        cell_id = h3lib.latlng_to_cell(float(lat), float(lon), h3_resolution)
        cell_to_poi_indices.setdefault(cell_id, []).append(i)

    held_out = []
    singletons = 0
    for cell_id, indices in cell_to_poi_indices.items():
        if len(indices) < 2:
            singletons += 1
            continue
        shuffled = rng.permutation(indices)
        n_test = max(1, int(np.ceil(test_ratio * len(indices))))
        held_out.extend(shuffled[:n_test].tolist())

    test_poi_mask = torch.zeros(N, dtype=torch.bool)
    if held_out:
        test_poi_mask[held_out] = True

    effective_ratio = test_poi_mask.sum().item() / N
    print(
        f"H3 POI split (res={h3_resolution}): {N} total POIs | "
        f"{test_poi_mask.sum().item()} held-out ({effective_ratio*100:.1f}%) | "
        f"{len(cell_to_poi_indices)} cells | {singletons} singleton cells (forced to train)"
    )
    return test_poi_mask


class LatLonTrajectoryDataset(Dataset):
    def __init__(self, csv_path, trajectory_ids, is_train=True, unique_coords_pool=None, bounds=None, pois_gdf=None, k=10, dist_threshold=250, mask_prob=0.15, h3_resolution=None, h3_cell_to_idx=None, regions_gdf=None, region_precomputed=None, test_poi_mask=None):
        """
        Args:
            csv_path (str): Path to the checkins CSV file.
            trajectory_ids (list): List of trajectory IDs assigned to this split.
            unique_coords_pool (torch.Tensor): Pre-computed unique coordinate lexicon shared across splits.
            bounds (dict): Pre-computed bounding box dict shared across splits.
            pois_gdf (GeoDataFrame, optional): OSM POI data for kNN lookup.
            mask_prob (float): Probability of masking a coordinate point during training.
            test_poi_mask (torch.BoolTensor, optional): Shape (N,). If provided and is_train=True,
                target positions whose POI is held-out (True) are replaced with -100 so
                CrossEntropyLoss ignores them. Has no effect when is_train=False.
        """
        self.is_train = is_train
        self.mask_prob = mask_prob
        
        df = pd.read_csv(csv_path, names=CSV_COLUMNS, header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # 1. Compute or inherit outlier-resilient spatial max/min boundaries (1st & 99th percentiles)
        if bounds is None:
            lat_quantiles = np.quantile(df['latitude'].values, [0.01, 0.99])
            lon_quantiles = np.quantile(df['longitude'].values, [0.01, 0.99])
            self.min_lat, self.max_lat = float(lat_quantiles[0]), float(lat_quantiles[1])
            self.min_lon, self.max_lon = float(lon_quantiles[0]), float(lon_quantiles[1])
        else:
            self.min_lat, self.max_lat = bounds['min_lat'], bounds['max_lat']
            self.min_lon, self.max_lon = bounds['min_lon'], bounds['max_lon']

        # Pre-compute neutral center point to act as the spatial mask token
        self.neutral_pad = torch.tensor(
            [(self.min_lat + self.max_lat) / 2.0, (self.min_lon + self.max_lon) / 2.0], 
            dtype=torch.float32
        )

        # 2. Extract unique coordinates vocabulary pool
        if unique_coords_pool is None:
            unique_df = df[['latitude', 'longitude']].drop_duplicates()
            self.unique_lat_lons = torch.tensor(unique_df.values, dtype=torch.float32)
        else:
            self.unique_lat_lons = unique_coords_pool

        # Map coordinate tuples to discrete index IDs for standard cross entropy classification
        coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(self.unique_lat_lons.tolist())}
        
        # POI Lookup Logic
        self.poi_knn_map = None
        if pois_gdf is not None:
            knn_df = get_topk_nearest_pois(self.unique_lat_lons, pois_gdf, k=k)
            knn_df.loc[knn_df['distance'] > dist_threshold, 'amenity'] = "<PAD>"
            amenity_to_idx = {cat: i for i, cat in enumerate(knn_df['amenity'].unique())}
            knn_df['amenity_idx'] = knn_df['amenity'].map(amenity_to_idx).fillna(-100).astype(int)
            self.poi_knn_map = torch.tensor(knn_df['amenity_idx'].values.reshape(-1, k), dtype=torch.long)
        
        # Filter for trajectories belonging to this structural partition split
        self.df = df[df['trail_id'].isin(trajectory_ids)].copy()
        self.df = self.df.sort_values(['trail_id', 'timestamp'])
        
        self.samples = []
        grouped = self.df.groupby('trail_id')
        for trail_id, group in grouped:
            if len(group) < 2:
                continue # Require at least 2 points for an autoregressive step

            coords_np = group[['latitude', 'longitude']].values.astype('float32')
            target_ids = [coord_to_idx.get(tuple(c), 0) for c in coords_np]

            if test_poi_mask is not None and is_train:
                for i, tid in enumerate(target_ids):
                    if tid >= 0 and test_poi_mask[tid]:
                        target_ids[i] = -100

            timestamps_unix = group['timestamp'].astype('int64').values // 10**9
            elapsed_min = (timestamps_unix - timestamps_unix[0]) / 60.0

            sample = {
                'coords': torch.tensor(coords_np, dtype=torch.float32),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'timestamps': torch.tensor(elapsed_min, dtype=torch.float32),
            }
            if self.poi_knn_map is not None:
                sample['poi_indices'] = self.poi_knn_map[target_ids]
            self.samples.append(sample)

        # H3 cell vocabulary — built from training unique coords, or inherited by test split
        self.h3_resolution = h3_resolution
        self.h3_cell_to_idx = None
        self.h3_cell_centroids = None
        self.unique_coord_h3_indices = None
        self.num_unique_h3_cells = 0

        if h3_resolution is not None:
            import h3 as h3lib

            if unique_coords_pool is None:
                # Training dataset: build H3 cell vocab from all unique training coordinates
                unique_coords_np = self.unique_lat_lons.numpy()
                raw_cell_ids = [h3lib.latlng_to_cell(float(lat), float(lon), h3_resolution)
                                for lat, lon in unique_coords_np]
                seen_cells = {}
                for cell_id in raw_cell_ids:
                    if cell_id not in seen_cells:
                        seen_cells[cell_id] = len(seen_cells)
                self.h3_cell_to_idx = seen_cells
                self.num_unique_h3_cells = len(seen_cells)

                centroids = [h3lib.cell_to_latlng(c) for c in seen_cells]
                self.h3_cell_centroids = torch.tensor(centroids, dtype=torch.float32)

                self.unique_coord_h3_indices = torch.tensor(
                    [seen_cells[c] for c in raw_cell_ids], dtype=torch.long
                )
                cell_to_idx_for_samples = seen_cells
            else:
                # Test dataset: reuse training vocab; unseen cells map to -1
                assert h3_cell_to_idx is not None, "test dataset requires h3_cell_to_idx from training dataset"
                self.h3_cell_to_idx = h3_cell_to_idx
                self.num_unique_h3_cells = len(h3_cell_to_idx)
                cell_to_idx_for_samples = h3_cell_to_idx

            # Attach per-sample H3 indices (vocab idx, or -1 for unseen cells)
            for sample in self.samples:
                coords_list = sample['coords'].numpy()
                h3_ids = [cell_to_idx_for_samples.get(
                    h3lib.latlng_to_cell(float(lat), float(lon), h3_resolution), -1
                ) for lat, lon in coords_list]
                sample['h3_ids'] = torch.tensor(h3_ids, dtype=torch.long)

        # Region vocab — built from OSM polygons via spatial join (training), or inherited (test)
        self.poi_region_indices = None
        self.poi_region_weights = None
        self.region_centroids   = None
        self.num_regions        = 0
        self.regions_max_R      = 1

        if regions_gdf is not None or region_precomputed is not None:
            import geopandas as gpd
            from shapely.geometry import Point

            if unique_coords_pool is None:
                # ── Training: spatial join to build region vocab ──────────────
                assert regions_gdf is not None
                unique_coords_np = self.unique_lat_lons.numpy()
                N = len(unique_coords_np)

                poi_gdf = gpd.GeoDataFrame(
                    {'poi_idx': np.arange(N)},
                    geometry=[Point(float(lon), float(lat)) for lat, lon in unique_coords_np],
                    crs='EPSG:4326',
                )
                regions_crs = regions_gdf.set_crs('EPSG:4326', allow_override=True)

                region_names = list(regions_crs['name'].values)
                region_to_idx = {name: i for i, name in enumerate(region_names)}
                num_regions = len(region_names)
                region_areas = regions_crs.geometry.area.values.astype(np.float64) + 1e-10

                joined = gpd.sjoin(poi_gdf, regions_crs[['name', 'geometry']],
                                   how='left', predicate='within')

                valid_joined = joined[joined['name'].notna()].copy()
                valid_joined = valid_joined[valid_joined['name'].isin(region_to_idx)].copy()
                valid_joined['r_idx'] = valid_joined['name'].map(region_to_idx).astype(int)

                max_R = 1
                if len(valid_joined) > 0:
                    max_R = max(int(valid_joined.groupby('poi_idx').size().max()), 1)

                poi_region_indices = torch.full((N, max_R), -1, dtype=torch.long)
                poi_region_weights = torch.zeros(N, max_R, dtype=torch.float32)

                for poi_idx, grp in valid_joined.groupby('poi_idx'):
                    r_idxs = grp['r_idx'].values
                    inv_areas = 1.0 / region_areas[r_idxs]
                    norm_w = inv_areas / inv_areas.sum()
                    k_r = len(r_idxs)
                    poi_region_indices[poi_idx, :k_r] = torch.tensor(r_idxs, dtype=torch.long)
                    poi_region_weights[poi_idx, :k_r] = torch.tensor(norm_w, dtype=torch.float32)

                centroids = [(geom.centroid.y, geom.centroid.x)
                             for geom in regions_crs.geometry]
                region_centroids = torch.tensor(centroids, dtype=torch.float32)

                # Fallback: POIs with no region → assign to nearest region centroid
                no_region = (poi_region_indices < 0).all(dim=-1)
                if no_region.any():
                    poi_coords_t = torch.tensor(unique_coords_np, dtype=torch.float32)
                    dists = torch.cdist(poi_coords_t[no_region], region_centroids)
                    nearest = dists.argmin(dim=-1)
                    poi_region_indices[no_region, 0] = nearest
                    poi_region_weights[no_region, 0] = 1.0
                    print(f"  Region fallback: {no_region.sum().item()} POIs assigned to nearest centroid")

                self.poi_region_indices = poi_region_indices
                self.poi_region_weights = poi_region_weights
                self.region_centroids   = region_centroids
                self.num_regions        = num_regions
                self.regions_max_R      = max_R
                self.region_to_idx      = region_to_idx

            else:
                # ── Test: inherit precomputed tensors from training ───────────
                assert region_precomputed is not None
                self.poi_region_indices = region_precomputed['poi_region_indices']
                self.poi_region_weights = region_precomputed['poi_region_weights']
                self.region_centroids   = region_precomputed['region_centroids']
                self.num_regions        = region_precomputed['num_regions']
                self.regions_max_R      = region_precomputed['max_R']

            # Attach per-sample region ids via target_ids (= unique POI index at each position)
            for sample in self.samples:
                poi_ids = sample['target_ids']          # (S,)
                clamped = poi_ids.clamp(min=0)
                r_ids = self.poi_region_indices[clamped].clone()   # (S, max_R)
                r_wts = self.poi_region_weights[clamped].clone()   # (S, max_R)
                invalid = poi_ids < 0
                r_ids[invalid] = -1
                r_wts[invalid] = 0.0
                sample['region_ids']     = r_ids
                sample['region_weights'] = r_wts

        print(f"📦 Loaded {'Train' if is_train else 'Test'} Split: {len(self.samples)} samples. Unique pool size: {len(self.unique_lat_lons)}"
              + (f" | H3 res={h3_resolution}, cells={self.num_unique_h3_cells}" if h3_resolution is not None else "")
              + (f" | Regions={self.num_regions}, max_R={self.regions_max_R}" if self.num_regions > 0 else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Bypass augmentation completely during verification/evaluation phases
        if not self.is_train or self.mask_prob <= 0.0:
            return sample
            
        # Clone tensors to avoid mutating static data structures across epochs
        coords = sample['coords'].clone()
        target_ids = sample['target_ids'].clone()

        seq_len = len(coords)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        if seq_len > 2:
            # Generate random masking assignments based on the configurable mask probability
            mask = torch.rand(seq_len) < self.mask_prob
            # Freeze index 0 as an unmasked spatial anchor to preserve baseline trajectory orientation
            mask[0] = False
            # Apply the neutral pad coordinate mask token to selected locations
            coords[mask] = self.neutral_pad

        augmented_sample = {
            'coords': coords,
            'target_ids': target_ids,
            'timestamps': sample['timestamps'],  # no masking needed
        }

        if 'poi_indices' in sample:
            augmented_sample['poi_indices'] = sample['poi_indices'].clone()

        if 'h3_ids' in sample:
            h3_ids = sample['h3_ids'].clone()
            h3_ids[mask] = -1  # masked positions signal "no cell" to the encoder
            augmented_sample['h3_ids'] = h3_ids

        if 'region_ids' in sample:
            region_ids = sample['region_ids'].clone()    # (S, max_R)
            region_wts = sample['region_weights'].clone()
            region_ids[mask] = -1
            region_wts[mask] = 0.0
            augmented_sample['region_ids']     = region_ids
            augmented_sample['region_weights'] = region_wts

        return augmented_sample


class LatLonCoordinateCollate:
    def __init__(self, padding_value_coords=-100.0, padding_value_targets=-100):
        self.padding_value_coords = padding_value_coords
        self.padding_value_targets = padding_value_targets

    def __call__(self, batch):
        out = {
            'coords':     pad_sequence([item['coords']     for item in batch], batch_first=True, padding_value=self.padding_value_coords),
            'targets':    pad_sequence([item['target_ids'] for item in batch], batch_first=True, padding_value=self.padding_value_targets),
            'lengths':    torch.tensor([len(item['coords']) for item in batch], dtype=torch.long),
            'timestamps': pad_sequence([item['timestamps'] for item in batch], batch_first=True, padding_value=0.0),
        }
        if 'h3_ids' in batch[0]:
            out['h3_ids'] = pad_sequence([item['h3_ids'] for item in batch], batch_first=True, padding_value=-1)
        if 'poi_indices' in batch[0]:
            out['poi_indices'] = pad_sequence([item['poi_indices'] for item in batch], batch_first=True, padding_value=-100)
        if 'region_ids' in batch[0]:
            out['region_ids']     = pad_sequence([item['region_ids']     for item in batch], batch_first=True, padding_value=-1)
            out['region_weights'] = pad_sequence([item['region_weights'] for item in batch], batch_first=True, padding_value=0.0)
        return out

def prepare_user_trajectories_and_split(csv_path, train_ratio=0.8, seed=42, max_len=128):
    """
    Restructures check-in data to group and concatenate trails by user,
    splitting them into sub-chunks of at most max_len without breaking 
    individual original trails in half. Saves to a unique temporary CSV.
    """
    # 1. Load data and clean missing coordinates
    df = pd.read_csv(csv_path, names=CSV_COLUMNS, header=0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # 2. Sort from oldest to newest globally across user timelines
    df = df.sort_values(by=['user_id', 'timestamp']).copy()
    
    # Keep track of the original trail boundaries before modification
    df['orig_trail_id'] = df['trail_id']
    
    # 3. Identify contiguous original trail blocks sequentially
    df['block_change'] = (df['orig_trail_id'] != df['orig_trail_id'].shift()) | (df['user_id'] != df['user_id'].shift())
    df['block_id'] = df['block_change'].cumsum()
    
    # Extract structural block properties
    block_info = df.groupby('block_id', sort=False).agg(
        user_id=('user_id', 'first'),
        row_count=('block_id', 'count')
    ).reset_index()
    
    # 4. Greedy Packing Engine: Allocate original trails to user chunks bounded by max_len
    new_trail_ids_map = {}
    current_user = None
    chunk_idx = 0
    current_chunk_len = 0
    
    for row in block_info.itertuples():
        b_id = row.block_id
        u_id = row.user_id
        count = row.row_count
        
        if u_id != current_user:
            current_user = u_id
            chunk_idx = 0
            current_chunk_len = 0
            
        if current_chunk_len + count > max_len and current_chunk_len > 0:
            chunk_idx += 1
            current_chunk_len = 0
            
        new_trail_ids_map[b_id] = f"{u_id}_{chunk_idx}"
        current_chunk_len += count
        
    df['trail_id'] = df['block_id'].map(new_trail_ids_map)
    block_info['new_trail_id'] = block_info['block_id'].map(new_trail_ids_map)
    
    # 5. Extract unique users and execute a random split at the user level
    unique_users = df['user_id'].unique()
    if seed is not None:
        np.random.seed(seed)
    shuffled_users = np.random.permutation(unique_users)
    
    split_idx = int(len(shuffled_users) * train_ratio)
    train_users = set(shuffled_users[:split_idx])
    test_users = set(shuffled_users[split_idx:])
    
    train_trail_ids = block_info[block_info['user_id'].isin(train_users)]['new_trail_id'].unique().tolist()
    test_trail_ids = block_info[block_info['user_id'].isin(test_users)]['new_trail_id'].unique().tolist()
    
    # 6. Generate unique temp filename with millisecond precision
    ms_timestamp = int(time.time() * 1000)
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_csv_path = os.path.join(tmp_dir, f"user_trajectories_{ms_timestamp}.csv")
    
    df = df.drop(columns=['orig_trail_id', 'block_change', 'block_id'])
    
    # 7. Persist to disk
    df.to_csv(tmp_csv_path, index=False)
    print(f"💾 Restructured dataset written to unique path: {tmp_csv_path}")
    print(f"👥 Users: {len(unique_users)} | Train Chunks: {len(train_trail_ids)} | Test Chunks: {len(test_trail_ids)}")
    
    return tmp_csv_path, train_trail_ids, test_trail_ids

def get_latlon_train_test_splits(csv_path, train_ratio=0.8):
    df = pd.read_csv(csv_path, names=CSV_COLUMNS, header=0, usecols=['trail_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    trajectory_starts = df.groupby('trail_id')['timestamp'].min().sort_values()
    sorted_trail_ids = trajectory_starts.index.tolist()
    
    split_idx = int(len(sorted_trail_ids) * train_ratio)
    train_ids = sorted_trail_ids[:split_idx]
    test_ids = sorted_trail_ids[split_idx:]
    
    return train_ids, test_ids