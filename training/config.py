import argparse

def get_unified_args():
    parser = argparse.ArgumentParser(description="Unified Configuration Engine for Geospatial Modeling Pipelines")
    
    # --- Environment & General Execution Context ---
    env_group = parser.add_argument_group("Environment Configurations")
    env_group.add_argument("--baseline", type=str, default="rff",
                           choices=["rff", "rff_sphere", "rff_advanced", "grid", "hex", "spherec", "discrete", "h3", "region", "mlp"],
                           help="Location encoder: RFF (2D or unit-sphere 3D), Space2Vec grid/hex, discrete POI, H3, region or coordinate MLP")
    env_group.add_argument("--save_path", type=str, default="save.pth", 
                           help="Checkpoint file for the selected model weights")
    env_group.add_argument("--gpu_id", type=int, default=0, 
                           help="CUDA device index; -1 selects CPU")
    env_group.add_argument("--seed", type=int, default=42,
                           help="Random seed for splits and torch/numpy")
    env_group.add_argument("--city", type=str, default="new_york",
                           help="MASSIVE-STEPS city directory name, such as new_york")

    # --- Shared Spatial Encoder Hyperparameters ---
    encoder_group = parser.add_argument_group("Spatial Encoder Configurations")
    encoder_group.add_argument("--d_model", type=int, default=256, 
                           help="Location embedding and transformer hidden dimension")
    encoder_group.add_argument("--rff_features", type=int, default=2048, 
                           help="Fourier feature count before the encoder MLP; paired RFF splits features equally between sine and cosine")
    encoder_group.add_argument("--min_scale", type=float, default=1.0, 
                           help="Smallest Gaussian frequency sampling scale in transformed coordinate units")
    encoder_group.add_argument("--max_scale", type=float, default=100.0, 
                           help="Largest Gaussian frequency sampling scale in transformed coordinate units")
    encoder_group.add_argument("--num_bands", type=int, default=32, 
                           help="Number of geometrically spaced frequency sampling scales between min_scale and max_scale")
    encoder_group.add_argument("--hidden_dim", type=int, default=2048, 
                           help="Hidden width of the location encoder MLP")
    encoder_group.add_argument("--pretrained_encoder_path", type=str, default=None, 
                           help="FileSystem pathway leading to pre-trained weights containing the proximity model spatial encoder state dict")
    encoder_group.add_argument("--freeze_encoder", action="store_true", 
                           help="Flag to freeze spatial batch norm and Fourier encoder features independently during optimization")
    encoder_group.add_argument("--num_layers_mlp", type=int, default=3,
                               help="Number of layers in the spatial encoder MLP projection head (0 bypasses via nn.Identity)")
    encoder_group.add_argument("--h3_resolution", type=int, default=9,
                               help="H3 hexagonal grid resolution level (0=coarsest, 15=finest). Only used with --baseline h3")
    encoder_group.add_argument("--h3_poi_split", action="store_true", default=False,
                               help="Use H3-stratified POI split: withholds test_ratio of POIs per H3 cell from training supervision")
    encoder_group.add_argument("--h3_split_resolution", type=int, default=9,
                               help="H3 resolution used to partition POIs for the H3-stratified split")
    encoder_group.add_argument("--h3_split_test_ratio", type=float, default=0.20,
                               help="Fraction of POIs per H3 cell to withhold from training supervision")
    encoder_group.add_argument("--region_admin_level", type=int, default=10,
                               help="OSM admin_level upper bound for region baseline (fetches levels 7-10). Only used with --baseline region")
    encoder_group.add_argument("--heuristic_gamma", type=float, default=1.5,
                               help="Distance-decay exponent γ for the gravity-history heuristic used in loss reweighting and ensemble eval")
    encoder_group.add_argument("--heuristic_beta", type=float, default=0.95,
                               help="History mixing weight β: 0=pure gravity, 1=pure history. Used in loss reweighting and ensemble eval")
    encoder_group.add_argument("--heuristic_alpha", type=float, default=0.6,
                               help="Ensemble mixing weight α at eval time: 0=pure model, 1=pure heuristic")
    encoder_group.add_argument("--heuristic_loss", action="store_true", default=False,
                               help="Train with heuristic residual loss (combined model+heuristic logits)")
    encoder_group.add_argument("--heuristic_popularity", action="store_true", default=True,
                               help="Include training-set popularity in the heuristic distance term. "
                                    "Enabled by default. The strict runner supports --no_heuristic_popularity.")

    # --- OSM Proximity & Cross-Modal CLIP Data Layout ---
    osm_group = parser.add_argument_group("OSM Proximity & CLIP Context Parameters")
    osm_group.add_argument("--dataset_length", type=int, default=1000000)
    osm_group.add_argument("--stash_dir", type=str, default="./stash")
    osm_group.add_argument("--max_dist_meters", type=float, default=1000.0)
    osm_group.add_argument("--num_samples_per_poi", type=int, default=20)
    osm_group.add_argument("--bert_dim", type=int, default=768)
    osm_group.add_argument("--textual_alignment_dim", type=int, default=256)
    osm_group.add_argument("--osm_batch_size", type=int, default=2048, 
                           help="Batch size explicitly dedicated to the OSM data stream updates")

    # --- Causal Sequence Trajectory Decoder Parameters ---
    trajectory_group = parser.add_argument_group("Causal Trajectory Transformer Configurations")
    trajectory_group.add_argument("--batch_size", type=int, default=512,
                           help="Mini-batch size processing execution constraints for sequential data")
    trajectory_group.add_argument("--backbone_type", type=str, default="qwen2",
                           choices=["qwen2", "gpt2", "qwen3"],
                           help="Sequential backbone architecture: qwen2 (RoPE+SwiGLU), gpt2 (learned PE+GELU), qwen3 (GQA+RoPE+SwiGLU)")
    trajectory_group.add_argument("--nhead", type=int, default=2)
    trajectory_group.add_argument("--num_layers", type=int, default=1)
    trajectory_group.add_argument("--dim_feedforward", type=int, default=512)
    trajectory_group.add_argument("--dropout", type=float, default=0.15)
    trajectory_group.add_argument("--clip_grad_norm", type=float, default=1.0)
    trajectory_group.add_argument("--pad_val_coords", type=float, default=-100.0)
    trajectory_group.add_argument("--pad_val_targets", type=int, default=-100)
    trajectory_group.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 50, 100])

    # --- Multi-Task Joint Optimization & Scheduling Hyperparameters ---
    opt_group = parser.add_argument_group("Optimization Controls")
    opt_group.add_argument("--epochs", type=int, default=30, 
                           help="Total execution trajectory model optimization run updates")
    opt_group.add_argument("--lr", type=float, default=4e-4)
    opt_group.add_argument("--early_stopping_patience", type=int, default=4)
    opt_group.add_argument("--weight_decay", type=float, default=0.01)
    opt_group.add_argument("--temperature", type=float, default=0.07)
    opt_group.add_argument("--mask_prob", type=float, default=0.0)
    opt_group.add_argument("--encoder_lr_mult", type=float, default=1.0,
                           help="LR multiplier for the location encoder (e.g. 0.1 when using a pretrained encoder)")
    opt_group.add_argument("--head_lr_mult", type=float, default=100.0,
                           help="LR multiplier for task-specific heads (alpha_head). Qwen trains at base LR.")
    opt_group.add_argument("--multistep_k", type=int, default=1,
                           help="Number of future steps to predict (1=standard next-token only)")
    opt_group.add_argument("--multistep_lambda", type=float, default=0.5,
                           help="Discount factor per additional prediction step")
    opt_group.add_argument("--use_time_fusion", action="store_true", default=False,
                           help="Enable temporal fusion: RoPE positions from real timestamps + Time2Vec delta-t encoder")
    opt_group.add_argument("--task", type=str, default="next_event",
                           choices=["next_event", "last_event"],
                           help="next_event: predict every next step; last_event: predict only the final trajectory event")
    opt_group.add_argument("--csv_path", type=str, default=None,
                           help="Absolute path to checkins CSV (overrides ./data/{city}/ convention)")
    opt_group.add_argument("--num_neg_samples", type=int, default=0,
                           help="Sampled negatives per batch (0=full softmax over all candidates). "
                                "Reduces GPU memory from O(B*S*N) to O(B*S*K). Requires multistep_k=1, "
                                "no heuristic_loss, and non-discrete baselines.")

    # Trajectory LR Milestones
    opt_group.add_argument("--lr_step1_epochs", type=int, default=10)
    opt_group.add_argument("--lr_step2_epochs", type=int, default=20)
    opt_group.add_argument("--lr_step2_mult", type=float, default=0.75)
    opt_group.add_argument("--lr_step3_mult", type=float, default=0.15)
    
    # Standalone OSM Scheduling (Used if running osm_train only)
    opt_group.add_argument("--osm_lr_step_size", type=int, default=10)
    opt_group.add_argument("--osm_lr_gamma", type=float, default=0.3)
    
    # Joint Training Switches
    opt_group.add_argument("--osm_warmup_epochs", type=int, default=5,
                           help="Number of initial epochs dedicated exclusively to optimizing the spatial encoder via OSM objectives")
    opt_group.add_argument("--osm_reg_weight", type=float, default=1.0,
                           help="Loss multiplier weight assigned to regularizing OSM multi-task objectives during joint training phases")
    opt_group.add_argument("--proximity_loss_weight", type=float, default=1.0,)
    opt_group.add_argument("--clip_loss_weight", type=float, default=1.0,)




    print(parser.parse_args())
    return parser.parse_args()
