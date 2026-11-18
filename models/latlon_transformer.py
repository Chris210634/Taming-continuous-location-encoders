import math
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, GPT2Config, GPT2Model

try:
    from transformers import Qwen3Config, Qwen3Model
    _QWEN3_AVAILABLE = True
except ImportError:
    _QWEN3_AVAILABLE = False


class Time2Vec(nn.Module):
    """
    Learnable time embedding: one linear term + (d_model-1) sinusoidal terms.
    Input t should be log1p-scaled elapsed minutes for numerical stability.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...) any shape — log1p-scaled elapsed minutes
        x = t.unsqueeze(-1) * self.w + self.b          # (..., d_model)
        return torch.cat([x[..., :1], torch.sin(x[..., 1:])], dim=-1)


class LatLonCausalTransformer(nn.Module):
    def __init__(self, location_encoder, d_model=256, nhead=4, num_layers=3, dim_feedforward=512,
                 dropout=0.1, max_len=2048, learnable_alpha=False, use_time_fusion=False,
                 backbone_type='qwen2'):
        super().__init__()

        self.location_encoder = location_encoder
        self.d_model = d_model
        self.learnable_alpha = learnable_alpha
        self.use_time_fusion = use_time_fusion
        self.backbone_type = backbone_type

        if learnable_alpha:
            self.alpha_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.alpha_head.weight)
            nn.init.zeros_(self.alpha_head.bias)  # sigmoid(0) = 0.5 at init

        if use_time_fusion:
            self.dt_encoder = Time2Vec(d_model)

        if backbone_type == 'qwen2':
            config = Qwen2Config(
                hidden_size=d_model,
                num_attention_heads=nhead,
                num_key_value_heads=nhead,
                num_hidden_layers=num_layers,
                intermediate_size=dim_feedforward,
                attention_dropout=dropout,
                max_position_embeddings=max_len,
                vocab_size=1,
            )
            self.transformer = Qwen2Model(config)

        elif backbone_type == 'gpt2':
            config = GPT2Config(
                n_embd=d_model,
                n_head=nhead,
                n_layer=num_layers,
                n_inner=dim_feedforward,
                attn_pdrop=dropout,
                resid_pdrop=dropout,
                embd_pdrop=0.0,
                vocab_size=1,
                n_positions=max_len,
            )
            self.transformer = GPT2Model(config)

        elif backbone_type == 'qwen3':
            if not _QWEN3_AVAILABLE:
                raise ImportError(
                    "Qwen3 not available in this transformers version. "
                    "Run: pip install --upgrade transformers"
                )
            config = Qwen3Config(
                hidden_size=d_model,
                num_attention_heads=nhead,
                num_key_value_heads=max(1, nhead // 4),  # GQA: 4x fewer KV heads
                num_hidden_layers=num_layers,
                intermediate_size=dim_feedforward,
                attention_dropout=dropout,
                max_position_embeddings=max_len,
                vocab_size=1,
            )
            self.transformer = Qwen3Model(config)

        else:
            raise ValueError(f"Unknown backbone_type '{backbone_type}'. Choose from: qwen2, gpt2, qwen3")

    def _base_hidden(self, lengths, embeddings, timestamps=None, x=None, neighborhood_poi=None):
        """
        Run transformer with temporal RoPE but WITHOUT Δt encoding.
        Used by multi-step training to apply a different per-k dt inside the loop.
        """
        if embeddings is not None:
            features = embeddings
            device = features.device
        else:
            features = self.location_encoder(x, neighborhood_poi=neighborhood_poi)
            device = features.device

        B, S, _ = features.shape
        lengths = lengths.to(device)
        attention_mask = (torch.arange(S, device=device).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.long)

        if self.backbone_type == 'gpt2':
            # GPT2 uses learned absolute PE; RoPE timestamp injection not applicable.
            # Use sequential position_ids [0, 1, ..., S-1] so GPT2's causal attention works correctly.
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            outputs = self.transformer(
                inputs_embeds=features,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            # Qwen2 / Qwen3: support timestamp-based RoPE position_ids
            if self.use_time_fusion and timestamps is not None:
                position_ids = timestamps.long().clamp(min=0).to(device)
            else:
                position_ids = None
            outputs = self.transformer(
                inputs_embeds=features,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return outputs.last_hidden_state  # (B, S, d_model)

    def apply_dt(self, hidden_states, dt):
        """
        Add Time2Vec Δt encoding to hidden states.
        dt: (B, S) elapsed minutes to the target k steps ahead, already clamp(min=0).
        """
        dt_emb = self.dt_encoder(torch.log1p(dt.clamp(min=0).to(hidden_states.device)))
        return hidden_states + dt_emb

    def forward(self, x=None, lengths=None, embeddings=None, neighborhood_poi=None, timestamps=None):
        """
        Args:
            x (Tensor, optional): Raw coordinates of shape (Batch_Size, Seq_Len, 2).
            lengths (Tensor): The actual unpadded lengths of the sequences.
            embeddings (Tensor, optional): Pre-computed embeddings of shape (Batch_Size, Seq_Len, d_model).
            timestamps (Tensor, optional): Elapsed minutes per position (Batch_Size, Seq_Len).
                When use_time_fusion=True, used as RoPE position_ids and for Δt encoding.
        """
        if x is not None and embeddings is not None:
            raise ValueError("Pass either raw coordinates 'x' or pre-computed 'embeddings', not both.")
        if x is None and embeddings is None:
            raise ValueError("You must provide either raw coordinates 'x' or pre-computed 'embeddings'.")

        hidden_states = self._base_hidden(lengths, embeddings, timestamps, x, neighborhood_poi)

        # Δt encoding: k=1 case — gap from position t to t+1.
        # Multi-step training bypasses forward() and calls _base_hidden + apply_dt directly
        # with the correct k-step dt per iteration.
        if self.use_time_fusion and timestamps is not None:
            timestamps = timestamps.to(hidden_states.device)
            dt = torch.zeros_like(timestamps)
            dt[:, :-1] = (timestamps[:, 1:] - timestamps[:, :-1]).clamp(min=0)
            hidden_states = self.apply_dt(hidden_states, dt)

        return hidden_states
    
# class LatLonCausalTransformer(nn.Module):
#     def __init__(self, location_encoder, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=2048):
#         super().__init__()
        
#         self.location_encoder = location_encoder
#         self.d_model = d_model
        
#         # 1. Generate the initial Sinusoidal Positional Encoding weights
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
#         # 2. Convert to a learnable parameter instead of a static buffer
#         # This automatically sets requires_grad=True and registers it with the optimizer
#         self.pe = nn.Parameter(pe)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation='gelu',
#             batch_first=True,
#             norm_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x=None, lengths=None, embeddings=None, neighborhood_poi=None):
#         """
#         Args:
#             x (Tensor, optional): Raw coordinates of shape (Batch_Size, Seq_Len, 2).
#             lengths (Tensor): The actual unpadded lengths of the sequences.
#             embeddings (Tensor, optional): Pre-computed embeddings of shape (Batch_Size, Seq_Len, d_model).
#         """
#         if x is not None and embeddings is not None:
#             raise ValueError("Pass either raw coordinates 'x' or pre-computed 'embeddings', not both.")
#         if x is None and embeddings is None:
#             raise ValueError("You must provide either raw coordinates 'x' or pre-computed 'embeddings'.")
            
#         if embeddings is not None:
#             features = embeddings
#             B, S, _ = features.shape
#             device = features.device
#         else:
#             B, S, _ = x.shape
#             device = x.device
#             features = self.location_encoder(x, neighborhood_poi=neighborhood_poi)
        
#         lengths = lengths.to(device)
        
#         # Slicing the parameter works exactly the same way as a buffer.
#         # PyTorch tracks gradients back to the sliced parameter weights.
#         features = features + self.pe[:, :S, :]
        
#         causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
#         key_padding_mask = torch.arange(S, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
                
#         hidden_states = self.transformer(
#             features, 
#             mask=causal_mask, 
#             src_key_padding_mask=key_padding_mask
#         )
#         return hidden_states