"""
Configuration for scMAGCL. Every hyperparameter used by main.py is read from here.
"""


def get_config():
    """Get default configuration"""
    config = {
        # Data augmentation
        'mask_prob': 0.25,      # Probability of masking each gene in the augmented view

        # Loss
        'lambda_recon': 1.0,    # Weight of the reconstruction (MSE) loss
        'lambda_contrast': 1.0, # Weight of the graph contrastive loss
        'alpha': 0.55,          # Weight of the original -> augmented contrastive term
        'beta': 0.4,            # Weight of the augmented -> original contrastive term
        'temperature': 0.5,     # Temperature of the contrastive loss

        # Training
        'lr': 0.001,
        'weight_decay': 1e-4,
        'epochs': 200,
        'batch_size': 128,      # Default; overridden by --batch_size
        'grad_clip': 1.0,       # Max gradient norm
        'test_size': 0.2,       # Fraction of cells held out for testing (stratified)
        'seed': 42,

        # Output
        'save_dir': 'results',  # Results are written to <save_dir>/<dataset name>/
    }

    return config
