"""Training-only coarse encoders for controlled strict-location comparisons."""
import numpy as np
import torch
from models.location_encoder import H3LocationEncoder


class StrictH3LocationEncoder(H3LocationEncoder):
    """Same H3 lookup baseline, with stable nearest-cell fallback and cell caching."""
    def _coords_to_h3_indices(self, coords):
        import h3
        if not hasattr(self, '_lookup_cache'):
            self._lookup_cache = dict(self.cell_to_idx)
            self._centroids64 = self.cell_centroids.detach().cpu().numpy().astype(np.float64)
        ids = []
        for lat, lon in coords.detach().cpu().numpy():
            cell = h3.latlng_to_cell(float(lat), float(lon), self.h3_resolution)
            if cell not in self._lookup_cache:
                center = np.asarray(h3.cell_to_latlng(cell), dtype=np.float64)
                self._lookup_cache[cell] = int(np.square(self._centroids64-center).sum(-1).argmin())
            ids.append(self._lookup_cache[cell])
        return torch.tensor(ids, device=coords.device, dtype=torch.long)


def build_strict_h3(train_coords, d_model, resolution, device):
    import h3
    cells = [h3.latlng_to_cell(float(lat),float(lon),resolution) for lat,lon in train_coords.cpu().tolist()]
    vocab = {c:i for i,c in enumerate(sorted(set(cells)))}
    indices = torch.tensor([vocab[c] for c in cells], dtype=torch.long, device=device)
    centroids = torch.tensor([h3.cell_to_latlng(c) for c in vocab],dtype=torch.float64,device=device)
    encoder = StrictH3LocationEncoder(len(vocab),d_model,resolution,centroids,vocab,indices).to(device)
    return encoder


def familiar_h3_cells(vocabulary, encoder):
    import h3
    return [h3.latlng_to_cell(float(lat),float(lon),encoder.h3_resolution) in encoder.cell_to_idx
            for lat,lon in vocabulary.tolist()]


class BoundedBandFourierFeatures(torch.nn.Module):
    """Paired sin/cos baseline with bounded, positive per-band frequency scales.

    At initialization, this exactly matches the fixed RFF features. Only band
    multipliers change: each remains in [1/limit, limit]. No test-based tuning.
    """
    def __init__(self, fixed_features, num_bands, limit=2.):
        super().__init__()
        if limit <= 1 or fixed_features.B.shape[1] % num_bands:
            raise ValueError('Need limit>1 and equally sized frequency bands')
        self.register_buffer('B', fixed_features.B.detach().clone())
        self.register_buffer('scale_factor', fixed_features.scale_factor.detach().clone())
        self.register_buffer('log_limit', torch.tensor(np.log(limit), dtype=self.B.dtype, device=self.B.device))
        self.raw_band_scales = torch.nn.Parameter(torch.zeros(num_bands,device=self.B.device,dtype=self.B.dtype))

    def multipliers(self):
        return torch.exp(self.log_limit * torch.tanh(self.raw_band_scales))

    def forward(self, coords):
        scales = self.multipliers().repeat_interleave(self.B.shape[1]//len(self.raw_band_scales))
        proj = coords @ (self.B * scales)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) * self.scale_factor
