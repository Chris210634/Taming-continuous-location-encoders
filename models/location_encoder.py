import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

def latlon_to_unit_sphere(coords):
    """(…, 2) latitude/longitude in degrees -> (…, 3) unit-sphere Cartesian x, y, z."""
    lat = torch.deg2rad(coords[..., 0])
    lon = torch.deg2rad(coords[..., 1])
    cos_lat = torch.cos(lat)
    return torch.stack([cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)], dim=-1)


def warm_start_spatial_bn(raw_encoder, dataloader,
                          device, num_batches=None, pad_val_coords=None):
    bn_layer = raw_encoder.spatial_batch_norm

    raw_encoder.eval()
    bn_layer.train()

    bn_layer.reset_running_stats()
    old_momentum = bn_layer.momentum
    bn_layer.momentum = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            coords = batch['coords'].to(device)
            assert coords.shape[-1] == 2
            if pad_val_coords is not None:
                valid_mask = (coords[:, :, 0] != pad_val_coords)
                coords = coords[valid_mask]

            if coords.size(0) > 0:
                if getattr(raw_encoder, 'spherical', False):
                    coords = latlon_to_unit_sphere(coords)
                bn_layer(coords)

    bn_layer.momentum = old_momentum
    bn_layer.eval()
    
class RandomFourierFeatures(nn.Module):
    """
    Projects low-dimensional coordinates into a multi-scale sinusoidal feature space.
    By partitioning frequencies into geometrically spaced bands, it captures coarse 
    macro patterns (city-scale) and crisp micro details (street-scale) simultaneously.
    """
    def __init__(self, in_features=2, num_features=128, min_scale=0.5, max_scale=20.0, num_bands=4):
        super().__init__()
        
        num_frequencies = num_features // 2
        if num_frequencies % num_bands != 0:
            raise ValueError(f"num_frequencies ({num_frequencies}) must be perfectly divisible by num_bands ({num_bands})")
            
        freqs_per_band = num_frequencies // num_bands
        scales = np.geomspace(min_scale, max_scale, num_bands)
        
        projection_bands = []
        for scale in scales:
            B_band = torch.randn(in_features, freqs_per_band) * scale
            projection_bands.append(B_band)
            
        B_master = torch.cat(projection_bands, dim=1)  # Shape: (in_features, num_frequencies)
        self.register_buffer('B', B_master)
        self.register_buffer('scale_factor', torch.tensor(np.sqrt(2.0 / num_features)))

    def forward(self, x):
        proj = torch.matmul(x, self.B)  # Shape: (M, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) * self.scale_factor  # Shape: (M, 2 * num_frequencies)

class RandomFourierFeaturesAdvanced(nn.Module):
    """
    Learnable Multi-Scale RFF Encoder.
    The frequency projection matrix and phase shifts are now nn.Parameters,
    allowing the model to optimize its own spatial resolution during training.
    """
    def __init__(
        self, 
        in_features=2, 
        num_features=128, 
        min_scale=0.5,
        max_scale=20.0, 
        num_bands=4,
        device="cuda"
    ):
        super().__init__()
        
        num_frequencies = num_features
        if num_frequencies % num_bands != 0:
            raise ValueError(f"num_frequencies ({num_frequencies}) must be divisible by num_bands ({num_bands})")
            
        self.num_frequencies = num_frequencies
        self.freqs_per_band = num_frequencies // num_bands
        scales = np.geomspace(min_scale, max_scale, num_bands)
        
        # 1. Initialize Projection Matrix (B) with geometric spacing
        projection_bands = []
        for scale in scales:
            # Initialize with Gaussian noise scaled by the geometric factor
            B_band = torch.randn(in_features, self.freqs_per_band) * scale
            projection_bands.append(B_band)
        
        B_master = torch.cat(projection_bands, dim=1) 
        
        # Make B a learnable parameter
        self.B = nn.Parameter(B_master)
        
        # 2. Make phase shift learnable
        shift = torch.rand(num_frequencies) * 2 * np.pi
        self.shift = nn.Parameter(shift)
        
        # 3. Normalization constant (kept as buffer as it is a hyperparameter)
        self.register_buffer('scale_factor', torch.tensor(np.sqrt(2.0 / num_frequencies)))

    def forward(self, x):
        """
        x: (Batch, N, in_features)
        Returns: (Batch, N, num_frequencies)
        """
        # Projection and shift are now using learnable parameters
        # The model will optimize the frequency weights (B) and phase shifts (shift)
        # to minimize the loss for your specific spatial distribution
        proj = torch.matmul(x, self.B) + self.shift
        
        return torch.cos(proj) * self.scale_factor
    
class Sphere2VecSphereC(nn.Module):
    """Sphere2Vec sphereC multi-scale location encoding (Mai et al., 2023b),
    vectorized from TorchSpatial's SphereSpatialRelationPositionEncoder
    (Wu et al., 2024). Per frequency f: [sin(lat_f), cos(lat_f)cos(lon_f),
    cos(lat_f)sin(lon_f)] on radian coordinates scaled by f, so inner products
    respect spherical geometry rather than planar lat/lon distance.

    Operates on RAW lat/lon degrees (no standardization, which would destroy
    the spherical geometry). The frequency for band scale w is w / sigma_rad,
    where sigma_rad is the training coordinate std in radians, making each
    (min_scale, max_scale) band span the same physical wavelengths as the
    standardized-coordinate encoders (Appendix B conversion).
    """

    def __init__(self, num_features=1920, min_scale=0.5, max_scale=20.0):
        super().__init__()
        if num_features % 3:
            raise ValueError("sphereC requires num_features divisible by 3")
        self.frequency_num = num_features // 3
        scales = np.geomspace(min_scale, max_scale, self.frequency_num)
        self.register_buffer("omegas", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("scale_factor", torch.tensor(np.sqrt(2.0 / num_features)))

    def forward(self, coords_deg, sigma_rad):
        # coords_deg: (M, 2) raw [lat, lon] degrees; sigma_rad: scalar tensor
        rad = torch.deg2rad(coords_deg)                      # (M, 2)
        freqs = (self.omegas / sigma_rad).view(1, 1, -1)     # (1, 1, F)
        angles = rad.unsqueeze(-1) * freqs                   # (M, 2, F)
        lat, lon = angles[:, 0, :], angles[:, 1, :]          # (M, F)
        cos_lat = torch.cos(lat)
        embeds = torch.stack([torch.sin(lat), cos_lat * torch.cos(lon),
                              cos_lat * torch.sin(lon)], dim=-1)  # (M, F, 3)
        return embeds.flatten(start_dim=1) * self.scale_factor   # (M, 3F)


class Space2VecFeaturesGrid(nn.Module):
    """
    Optimized GridCellSpatialRelationPositionEncoder.
    Uses min_scale/max_scale to define frequency bands for consistency.
    """
    def __init__(
        self,
        coord_dim=2,
        num_features=16,
        min_scale=0.5,
        max_scale=20.0,
        freq_init="geometric",
        device="cuda",
    ):
        super().__init__()
        self.coord_dim = coord_dim
        assert num_features % 4 == 0, "num_features must be divisible by 4"
        self.frequency_num = num_features // 4
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.device = device

        freqs = self._calculate_frequencies(freq_init)
        self.register_buffer("freqs", freqs)
        self.pos_enc_output_dim = int(self.coord_dim * self.frequency_num * 2)
        self.register_buffer('scale_factor', torch.tensor(np.sqrt(2.0 / num_features)))

    def _calculate_frequencies(self, freq_init):
        if freq_init == "geometric":
            # Geometric spacing between min_scale and max_scale
            log_increment = math.log(float(self.max_scale) / float(self.min_scale)) / (self.frequency_num - 1)
            freqs = self.min_scale * torch.exp(
                torch.arange(self.frequency_num, dtype=torch.float32) * log_increment
            )
            return freqs
        else:
            raise NotImplementedError(f"Frequency initialization '{freq_init}' not implemented.")

    def forward(self, coords):
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32).to(self.device)
        coords = coords.unsqueeze(-1)
        freqs = self.freqs.view(1, 1, 1, -1)
        angles = coords * freqs
        embeds = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        batch_size, num_pts, _, _, _ = embeds.shape
        return embeds.view(batch_size, num_pts, -1) * self.scale_factor

class Space2VecFeaturesHex(nn.Module):
    """
    Optimized Theory-based Grid Cell Spatial Encoder.
    Uses min_scale/max_scale to define frequency bands for consistency.
    """
    def __init__(
        self,
        coord_dim=2,
        num_features=16,
        min_scale=0.5,
        max_scale=20.0,
        freq_init="geometric",
        device="cuda",
    ):
        super().__init__()
        self.coord_dim = coord_dim
        assert num_features % 6 == 0, "num_features must be divisible by 6"
        self.frequency_num = num_features // 6
        self.device = device
        
        self.register_buffer("freqs", self._calculate_frequencies(freq_init, min_scale, max_scale))

        sqrt3 = math.sqrt(3)
        unit_vecs = torch.tensor([
            [1.0, 0.0],
            [-0.5, sqrt3 / 2.0],
            [-0.5, -sqrt3 / 2.0]
        ], dtype=torch.float32)
        self.register_buffer("unit_vecs", unit_vecs)
        self.pos_enc_output_dim = int(6 * self.frequency_num)
        self.register_buffer('scale_factor', torch.tensor(np.sqrt(2.0 / num_features)))

    def _calculate_frequencies(self, freq_init, min_scale, max_scale):
        if freq_init == "geometric":
            log_increment = math.log(float(max_scale) / float(min_scale)) / (self.frequency_num - 1)
            freqs = min_scale * torch.exp(
                torch.arange(self.frequency_num, dtype=torch.float32) * log_increment
            )
            return freqs
        raise NotImplementedError("Only 'geometric' init is implemented.")

    def forward(self, coords):
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32).to(self.device)
        projections = torch.matmul(coords, self.unit_vecs.T)
        angles = projections.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        interleaved = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        return interleaved.permute(0, 1, 3, 2, 4).flatten(start_dim=2) * self.scale_factor
    
class LocationEncoder(nn.Module):
    def __init__(
        self, 
        feature_type='rff', 
        out_dim=256, 
        num_features=128, 
        min_scale=0.5, 
        max_scale=20.0, 
        num_bands=4, 
        hidden_dim=256,
        num_amenities=None,
        device="cuda",
        num_layers_mlp=3
    ):
        super().__init__()
        self.feature_type = feature_type.lower()
        # 'rff_sphere' converts lat/lon degrees to unit-sphere x,y,z before normalization,
        # so Euclidean distance in encoder input space is monotone in geodesic distance.
        self.spherical = (self.feature_type == 'rff_sphere')
        coord_dim = 3 if self.spherical else 2
        self.spatial_batch_norm = nn.BatchNorm1d(num_features=coord_dim)
        self.use_poi = (num_amenities is not None)

        # 1. Instantiate encoder
        if self.feature_type in ('rff', 'rff_sphere'):
            self.encoder = RandomFourierFeatures(in_features=coord_dim, num_features=num_features, min_scale=min_scale, max_scale=max_scale, num_bands=num_bands)
            mlp_in = num_features
        elif self.feature_type == 'rff_advanced':
            self.encoder = RandomFourierFeaturesAdvanced(in_features=2, num_features=num_features, min_scale=min_scale, max_scale=max_scale, num_bands=num_bands, device=device)
            mlp_in = num_features
        elif self.feature_type == 'grid':
            self.encoder = Space2VecFeaturesGrid(coord_dim=2, num_features=num_features, min_scale=min_scale, max_scale=max_scale, device=device)
            mlp_in = num_features
        elif self.feature_type == 'hex':
            self.encoder = Space2VecFeaturesHex(coord_dim=2, num_features=num_features, min_scale=min_scale, max_scale=max_scale, device=device)
            mlp_in = num_features
        elif self.feature_type == 'spherec':
            # Sphere2Vec-sphereC consumes raw lat/lon degrees; the batch-norm layer
            # is warm-started only to record coordinate statistics (sigma for the
            # wavelength conversion) and is never applied to the inputs.
            self.encoder = Sphere2VecSphereC(num_features=num_features, min_scale=min_scale, max_scale=max_scale)
            mlp_in = num_features
        elif self.feature_type == 'mlp':
            self.encoder = nn.Identity()
            mlp_in = 2
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        # 2. Setup Fusion (Must match mlp_in)
        if self.use_poi:
            self.poi_embedding = nn.Embedding(num_amenities, mlp_in, padding_idx=-100)
        
        # 3. Dynamic MLP Construction Engine
        if num_layers_mlp == 0:
            if mlp_in != out_dim:
                raise ValueError(
                    f"num_layers_mlp=0 uses nn.Identity, but encoder output dim ({mlp_in}) "
                    f"!= out_dim ({out_dim}). Use num_layers_mlp=1 for a linear projection."
                )
            self.mlp = nn.Identity()
        elif num_layers_mlp == 1:
            self.mlp = nn.Linear(mlp_in, out_dim)
        else:
            layers = []
            # Input layer mapping to inner hidden capacity width
            layers.append(nn.Linear(mlp_in, hidden_dim))
            layers.append(nn.ReLU())
            
            # Dynamic hidden deep structural layers
            for _ in range(num_layers_mlp - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                
            # Final projection mapping to structural transformer boundary width
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, coords, neighborhood_poi=None, targets=None):
        orig_shape = coords.shape
        if len(orig_shape) == 3:
            B, S, _ = orig_shape
            coords = coords.reshape(-1, 2)

        if self.feature_type == 'spherec':
            # Raw-degree spherical encoding; BN supplies only the recorded coordinate
            # std for the wavelength conversion (never applied to the inputs).
            sigma_rad = torch.deg2rad(self.spatial_batch_norm.running_var.mean().sqrt())
            features = self.encoder(coords, sigma_rad)
        else:
            if self.spherical:
                coords = latlon_to_unit_sphere(coords)
            normalized_coords = self.spatial_batch_norm(coords)
            features = self.encoder(normalized_coords) if self.feature_type != 'mlp' else normalized_coords
        
        if len(features.shape) > 2:
            features = features.squeeze(0)
            
        # Additive Fusion
        if self.use_poi:
            assert neighborhood_poi is not None
            # Flatten indices to (M*K)
            poi_embs = self.poi_embedding(neighborhood_poi.reshape(-1, neighborhood_poi.shape[-1]))
            # Mean pool over K neighbors: (M, mlp_in)
            poi_context = poi_embs.mean(dim=1)
            # Additive fusion
            features = features + poi_context
            
        out = self.mlp(features)
        
        if len(orig_shape) == 3:
            out = out.reshape(B, S, -1)
            
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteLocationEncoder(nn.Module):
    """
    A discrete location encoder baseline that maps raw continuous coordinates 
    to embeddings. Uses an identical BatchNorm1d approach to the continuous encoder
    and implements a hard 1-NN Dirac delta assignment to handle unseen coordinates.
    """
    def __init__(self, num_unique, d_model, unique_coords, init_gamma=1e8):
        super().__init__()
        
        # Identical batch norm initialization
        self.spatial_batch_norm = nn.BatchNorm1d(num_features=2)
        self.embedding = nn.Embedding(num_unique, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        self.register_buffer("unique_coords", unique_coords.clone())
        
        # Kept as buffer (not parameter) for state-dict checkpoint compatibility only — never used in forward.
        self.register_buffer('log_gamma', torch.log(torch.tensor([init_gamma], dtype=torch.float32)))

    def forward(self, coords, targets=None, **_):
        # coords can be (M, 2) during distillation or (B, S, 2) during transformer sequence tracking
        orig_shape = coords.shape
        
        # Identical 3D flattening check to the continuous LocationEncoder
        if len(orig_shape) == 3:
            B, S, _ = orig_shape
            coords = coords.reshape(-1, 2)
            if targets is not None:
                targets = targets.reshape(-1)
        
        if targets is not None:
            assert self.training
            clamped = targets.clamp(min=0)
            out = self.embedding(clamped)
            # Detect augmentation-masked positions: real target indices (>= 0) whose input
            # coord was replaced with neutral_pad. The coord no longer matches the coordinate
            # stored in unique_coords at that target index, so we zero the embedding out to
            # be consistent with what continuous encoders produce at masked positions.
            is_real = targets >= 0
            if is_real.any():
                expected = self.unique_coords[clamped[is_real]]
                is_aug_masked = ~torch.isclose(coords[is_real], expected, atol=1e-5).all(dim=-1)
                if is_aug_masked.any():
                    aug_mask = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
                    aug_mask[is_real.nonzero(as_tuple=True)[0][is_aug_masked]] = True
                    out = out.masked_fill(aug_mask.unsqueeze(-1), 0.0)

        else:
            assert not self.training
            # --- EVALUATION MODE: Hard Dirac Delta Selector (1-NN) ---
            # Apply normalization identically to protect coordinate distributions
            normalized_coords = self.spatial_batch_norm(coords)
            
            # Safely normalize known unique reference coordinates without modifying training running statistics
            was_training = self.spatial_batch_norm.training
            self.spatial_batch_norm.eval()
            unique_coords_norm = self.spatial_batch_norm(self.unique_coords)
            if was_training:
                self.spatial_batch_norm.train()
                
            # Compute squared Euclidean distance matrix between inputs and unique nodes
            dist_squared = torch.cdist(normalized_coords, unique_coords_norm) ** 2
            
            # 🎯 The Dirac Delta Limit: Find the absolute closest known POI index
            closest_poi_indices = torch.argmin(dist_squared, dim=-1)
            
            # Directly extract the pristine embedding of the closest geographic neighbor
            out = self.embedding(closest_poi_indices)

        # Identical structural reconstruction check to the continuous LocationEncoder
        if len(orig_shape) == 3:
            out = out.reshape(B, S, -1)

        return out


class H3LocationEncoder(nn.Module):
    """
    Discrete location encoder using Uber's H3 hexagonal hierarchical grid.
    All coordinates within the same H3 cell share a single learned embedding.
    Vocabulary is built from training data only. Unseen cells at eval time fall
    back to the nearest training-cell centroid (1-NN in lat/lon space).
    No MLP — outputs directly at d_model dimensionality.

    Training path: precomputed h3_indices (vocab idx per position) are passed in
    from the dataloader. Positions with index -1 (masked or truly unseen) are
    zeroed out.

    Eval path: h3 library converts raw coords to cell IDs; the cell_to_idx dict
    maps IDs to vocab indices; unseen cells get 1-NN fallback.

    Note: cell_to_idx is a plain Python dict and is NOT persisted in the state
    dict. Reload the encoder from the dataset when loading a checkpoint.
    """

    def __init__(self, num_unique_cells, d_model, h3_resolution,
                 cell_centroids, cell_to_idx, unique_coord_h3_indices):
        """
        Args:
            num_unique_cells: number of unique H3 cells in the training vocab
            d_model: embedding dimension
            h3_resolution: H3 resolution (0–15)
            cell_centroids: (num_unique_cells, 2) float32 tensor of cell lat/lon centroids
            cell_to_idx: dict mapping H3 cell ID string -> vocab index
            unique_coord_h3_indices: (num_unique_training_coords,) int64 tensor —
                maps each unique training coordinate index to its H3 vocab index
        """
        super().__init__()
        self.h3_resolution = h3_resolution
        self.d_model = d_model
        self.cell_to_idx = cell_to_idx  # not in state_dict; reconstruct from dataset on reload

        self.embedding = nn.Embedding(num_unique_cells, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        self.register_buffer('cell_centroids', cell_centroids)
        self.register_buffer('unique_coord_h3_indices', unique_coord_h3_indices)

    def _coords_to_h3_indices(self, coords):
        """Convert (M, 2) lat/lon tensor to vocab indices, with 1-NN fallback for unseen cells."""
        import h3 as h3lib
        coords_np = coords.detach().cpu().numpy()
        idx_list = [
            self.cell_to_idx.get(h3lib.latlng_to_cell(float(lat), float(lon), self.h3_resolution), -1)
            for lat, lon in coords_np
        ]
        indices = torch.tensor(idx_list, dtype=torch.long, device=coords.device)

        unseen = indices < 0
        if unseen.any():
            # Use cell centroid (not raw coord) so all points in the same unseen hex
            # snap to the same nearest training cell → uniform embedding within each hex.
            unseen_cell_ids = [
                h3lib.latlng_to_cell(float(lat), float(lon), self.h3_resolution)
                for lat, lon in coords_np[unseen.cpu().numpy()]
            ]
            unseen_centroids = torch.tensor(
                [h3lib.cell_to_latlng(c) for c in unseen_cell_ids],
                dtype=torch.float32, device=coords.device,
            )
            dists = torch.cdist(unseen_centroids, self.cell_centroids)
            indices[unseen] = dists.argmin(dim=-1)

        return indices

    def forward(self, coords, h3_indices=None, **_):
        """
        Args:
            coords: (B, S, 2) or (M, 2) lat/lon tensor
            h3_indices: (B, S) or (M,) precomputed vocab indices from the dataloader.
                        Index -1 means masked/unseen — the encoder zeros that position.
                        If None, computes indices from coords via the h3 library (eval path).
            **kwargs: absorbs unused keyword args (e.g. targets= from callers that don't know
                      this is an H3 encoder)
        """
        orig_shape = coords.shape
        if len(orig_shape) == 3:
            B, S, _ = orig_shape
            coords = coords.reshape(-1, 2)
            if h3_indices is not None:
                h3_indices = h3_indices.reshape(-1)

        if h3_indices is not None:
            # Training path: use precomputed indices from the dataloader.
            is_invalid = h3_indices < 0
            out = self.embedding(h3_indices.clamp(min=0))
            if is_invalid.any():
                out = out.masked_fill(is_invalid.unsqueeze(-1), 0.0)
        else:
            # Eval path: derive indices from raw coords via h3 library.
            h3_indices = self._coords_to_h3_indices(coords)
            out = self.embedding(h3_indices)

        if len(orig_shape) == 3:
            out = out.reshape(B, S, -1)

        return out


class RegionLocationEncoder(nn.Module):
    """
    Location encoder based on OSM administrative / neighborhood boundary polygons.

    Each region gets a learned embedding.  A POI can belong to multiple overlapping
    regions; the output is an inverse-area-weighted mean of all member embeddings.
    Smaller (more-specific) regions receive higher weight.

    An extra "unknown" embedding covers POIs that belong to no region.

    Training path: precomputed (region_ids, region_weights) tensors from the
    dataloader are used directly — no geometry at forward time.

    Eval path (region_ids=None): raw coords are matched to the nearest known POI
    via 1-NN on the full unique-coordinate set, then that POI's region memberships
    are used.  This is exact for POIs seen during training and a sensible
    approximation for out-of-vocab coordinates.

    Note: region_centroids and poi_region_* tensors are NOT persisted in the
    state dict.  Rebuild the encoder from the dataset when loading a checkpoint.
    """

    def __init__(self, num_regions, d_model,
                 region_centroids,
                 poi_region_indices,
                 poi_region_weights,
                 unique_coords,
                 regions_gdf=None):
        """
        Args:
            num_regions: number of named OSM regions in the training vocab
            d_model: embedding dimension
            region_centroids: (num_regions, 2) float32 lat/lon tensor
            poi_region_indices: (num_unique_pois, max_R) long tensor; -1 = no region
            poi_region_weights: (num_unique_pois, max_R) float32 tensor; 0 = no region
            unique_coords: (num_unique_pois, 2) float32 lat/lon tensor
            regions_gdf: GeoDataFrame with 'name' and 'geometry' columns (EPSG:4326).
                         If provided, eval-path queries use point-in-polygon lookup
                         instead of 1-NN to training POIs.
        """
        super().__init__()
        self.d_model = d_model

        self.spatial_batch_norm = nn.BatchNorm1d(num_features=2)

        self.embedding = nn.Embedding(num_regions, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        self.register_buffer('region_centroids',    region_centroids)
        self.register_buffer('poi_region_indices',  poi_region_indices)
        self.register_buffer('poi_region_weights',  poi_region_weights)
        self.register_buffer('unique_coords',       unique_coords.clone())

        # Spatial index for eval-path point-in-polygon lookup (not in state_dict)
        self._region_tree  = None
        self._region_areas = None
        if regions_gdf is not None:
            from shapely.strtree import STRtree
            import numpy as np
            gdf = regions_gdf.set_crs('EPSG:4326', allow_override=True)
            self._region_areas = gdf.geometry.area.values.astype(np.float64) + 1e-10
            self._region_tree  = STRtree(gdf.geometry.values)

    def _poi_indices_for_coords(self, coords, chunk_size=1024):
        """Chunked 1-NN in batch-norm-normalized space → nearest unique-POI index."""
        was_training = self.spatial_batch_norm.training
        self.spatial_batch_norm.eval()
        uc_norm = self.spatial_batch_norm(self.unique_coords.float())
        if was_training:
            self.spatial_batch_norm.train()

        results = []
        for i in range(0, coords.shape[0], chunk_size):
            c_norm = self.spatial_batch_norm(coords[i:i + chunk_size].float())
            dists = torch.cdist(c_norm, uc_norm)
            results.append(dists.argmin(dim=-1))
        return torch.cat(results) if results else torch.empty(0, dtype=torch.long, device=coords.device)

    def _spatial_lookup_for_coords(self, coords):
        """
        Point-in-polygon lookup for each coordinate using the stored STRtree.
        Replicates the dataset's sjoin logic: inverse-area weights for all
        containing regions, fallback to nearest centroid for unmatched points.
        """
        import numpy as np
        from shapely.geometry import Point

        coords_np = coords.cpu().numpy()
        M = coords_np.shape[0]
        max_R = self.poi_region_indices.shape[1]
        device = coords.device

        r_ids_np = np.full((M, max_R), -1, dtype=np.int64)
        r_wts_np = np.zeros((M, max_R), dtype=np.float32)

        no_region = []
        for i, (lat, lon) in enumerate(coords_np):
            point = Point(float(lon), float(lat))
            idxs = self._region_tree.query(point, predicate='within')
            if len(idxs) > 0:
                idxs = idxs[:max_R]
                inv_areas = 1.0 / self._region_areas[idxs]
                norm_w = (inv_areas / inv_areas.sum()).astype(np.float32)
                r_ids_np[i, :len(idxs)] = idxs
                r_wts_np[i, :len(idxs)] = norm_w
            else:
                no_region.append(i)

        if no_region:
            fallback = torch.tensor(coords_np[no_region], dtype=torch.float32, device=self.region_centroids.device)
            dists = torch.cdist(fallback, self.region_centroids)
            nearest = dists.argmin(dim=-1).cpu().numpy()
            for j, i in enumerate(no_region):
                r_ids_np[i, 0] = nearest[j]
                r_wts_np[i, 0] = 1.0

        r_ids = torch.tensor(r_ids_np, dtype=torch.long, device=device)
        r_wts = torch.tensor(r_wts_np, dtype=torch.float32, device=device)
        return r_ids, r_wts

    def _weighted_embed(self, r_ids, r_wts):
        """
        Inverse-area weighted mean of member region embeddings.
        Every position is guaranteed to have at least one valid region (dataset
        assigns unmatched POIs to their nearest region centroid at build time).

        Args:
            r_ids: (..., max_R) long — region vocab indices, -1 = absent slot
            r_wts: (..., max_R) float — normalised inverse-area weights
        Returns:
            (..., d_model)
        """
        is_valid = r_ids >= 0                                # (..., max_R)
        embs = self.embedding(r_ids.clamp(min=0))           # (..., max_R, d_model)
        embs = embs * is_valid.unsqueeze(-1).float()        # zero absent slots
        return (embs * r_wts.unsqueeze(-1)).sum(dim=-2)     # (..., d_model)

    def forward(self, coords, region_ids=None, region_weights=None, **_):
        """
        Args:
            coords: (B, S, 2) or (M, 2) lat/lon
            region_ids: (B, S, max_R) or (M, max_R) — precomputed from dataloader
            region_weights: same shape as region_ids — normalised inverse-area weights
            **_: absorbs unused kwargs (e.g. targets=)
        """
        orig_shape = coords.shape
        if len(orig_shape) == 3:
            B, S, _ = orig_shape
            coords = coords.reshape(-1, 2)
            if region_ids is not None:
                region_ids     = region_ids.reshape(-1, region_ids.shape[-1])
                region_weights = region_weights.reshape(-1, region_weights.shape[-1])

        if region_ids is not None:
            # Training / supervised eval: use precomputed memberships
            out = self._weighted_embed(region_ids, region_weights)
        elif self._region_tree is not None:
            # Eval with geometry: exact point-in-polygon lookup + centroid fallback
            r_ids, r_wts = self._spatial_lookup_for_coords(coords)
            out = self._weighted_embed(r_ids, r_wts)
        else:
            # Eval without geometry: 1-NN to nearest known training POI
            poi_idx = self._poi_indices_for_coords(coords)
            r_ids = self.poi_region_indices[poi_idx]
            r_wts = self.poi_region_weights[poi_idx]
            out = self._weighted_embed(r_ids, r_wts)

        if len(orig_shape) == 3:
            out = out.reshape(B, S, -1)

        return out