"""
Train scMAGCL on one dataset and evaluate clustering on held-out cells.

Usage:
    python main.py --dataset ./data/pollen.h5 --batch_size 128
"""

import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from config import get_config
from utils import (
    setup_seed,
    create_augmented_data,
    scRNASeqDataset,
    visualize_and_evaluate
)
from model import scMAGCL, final_cl_loss


# Depths at which the multi-level graph contrastive loss is applied:
# latent space, decoder hidden layer, reconstruction
CL_LAYERS = ['h2_128', 'd1_512', 'recon']


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(path):
    """Load an .h5 file with the expression matrix X (cells x genes) and labels Y"""
    with h5py.File(path, 'r') as f:
        X = f['X'][()].astype(np.float32)
        y = f['Y'][()].ravel()

    # Works for integer or string labels; maps them to 0..k-1
    y = LabelEncoder().fit_transform(y)
    return X, y


# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, test_loader, config, device):
    """
    Train scMAGCL and return the test-set embedding from the epoch with the
    lowest test reconstruction loss
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )

    mse_criterion = nn.MSELoss()

    best_loss = float('inf')
    best_latent = None
    best_labels = None

    print("\n" + "="*70)
    print("Training scMAGCL")
    print("="*70)

    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()

        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_contrast_loss = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_x_aug = create_augmented_data(batch_x, mask_prob=config['mask_prob'])

            optimizer.zero_grad()

            outputs = model(batch_x, batch_x_aug)
            ae1_out = outputs['ae1_out']
            ae2_out = outputs['ae2_out']

            # Reconstruct the original matrix from the augmented view
            recon_loss = config['lambda_recon'] * mse_criterion(ae2_out['recon'], batch_x)

            # Graph contrastive loss at each depth, using the intra-cell graphs
            # (self-attention) and inter-cell graphs (cross-attention) of that layer
            contrast_loss = 0
            for i, layer in enumerate(CL_LAYERS):
                contrast_loss += final_cl_loss(
                    config['alpha'],
                    config['beta'],
                    ae1_out[layer],
                    ae2_out[layer],
                    outputs['graphs_ae1'][i],   # intra-cell graph, original view
                    outputs['graphs_ae2'][i],   # intra-cell graph, augmented view
                    outputs['graphs_ae12'][i],  # inter-cell graph, original -> augmented
                    outputs['graphs_ae21'][i],  # inter-cell graph, augmented -> original
                    config['temperature']
                )

            weighted_contrast_loss = config['lambda_contrast'] * contrast_loss

            (recon_loss + weighted_contrast_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_contrast_loss += contrast_loss.item()
            epoch_total_loss += recon_loss.item() + weighted_contrast_loss.item()

        scheduler.step()

        # Evaluation on held-out cells
        model.eval()
        test_loss = 0
        all_latent = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_x_aug = create_augmented_data(batch_x, mask_prob=config['mask_prob'])

                outputs = model(batch_x, batch_x_aug)
                ae1_out = outputs['ae1_out']

                test_loss += mse_criterion(ae1_out['recon'], batch_x).item()

                # 128-dim latent embedding of the original view
                all_latent.append(ae1_out['h2_128'].cpu().numpy())
                all_labels.append(batch_y.numpy())

        if test_loss < best_loss:
            best_loss = test_loss
            best_latent = np.vstack(all_latent)
            best_labels = np.hstack(all_labels)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Total Loss={epoch_total_loss:.4f} "
                  f"(Recon={epoch_recon_loss:.4f}, "
                  f"Contrast={epoch_contrast_loss:.4f}), "
                  f"Test Loss={test_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print("="*70)

    return best_latent, best_labels, total_time


# ============================================================================
# Main Execution
# ============================================================================

def main(dataset_path, batch_size=None):
    """
    Train scMAGCL on one dataset and evaluate clustering on held-out cells

    Args:
        dataset_path: Path to an .h5 file with keys X and Y
        batch_size: Batch size for training; falls back to config's default
    """

    config = get_config()
    if batch_size is not None:
        config['batch_size'] = batch_size

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_dir = os.path.join(config['save_dir'], dataset_name)

    print(f"Random Seed: {config['seed']}")
    setup_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading dataset: {dataset_name}")
    X, y = load_dataset(dataset_path)
    n_clusters = len(np.unique(y))
    print(f"Cells: {X.shape[0]}, Genes: {X.shape[1]}, Cell types: {n_clusters}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['seed'], stratify=y
    )

    train_loader = DataLoader(scRNASeqDataset(X_train, y_train),
                              batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(scRNASeqDataset(X_test, y_test),
                             batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = scMAGCL(input_dim=X.shape[1]).to(device)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    best_latent, best_labels, training_time = train_model(
        model, train_loader, test_loader, config, device
    )

    # K-means on the test-set embedding; saves cluster labels and UMAP plot
    acc, nmi, ari = visualize_and_evaluate(
        latent=best_latent,
        labels=best_labels,
        n_clusters=n_clusters,
        seed=config['seed'],
        save_dir=save_dir
    )

    # Append this run to the results table
    results_path = os.path.join(config['save_dir'], 'results.csv')
    row = pd.DataFrame([{
        'Dataset': dataset_name,
        'Seed': config['seed'],
        'Batch_size': config['batch_size'],
        'Epochs': config['epochs'],
        'Cells': X.shape[0],
        'Genes': X.shape[1],
        'Clusters': n_clusters,
        'ACC': acc,
        'NMI': nmi,
        'ARI': ari,
        'Train_time_s': training_time,
    }])
    row.to_csv(results_path, mode='a', index=False,
               header=not os.path.exists(results_path))

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Clustering Accuracy (ACC): {acc:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Results saved to: {save_dir} and {results_path}")
    print("="*70)

    return model, acc, nmi, ari


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train scMAGCL on a dataset and evaluate clustering"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Path to an .h5 file with keys X (cells x genes) and Y (cell labels)",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help="Training batch size (default: value in config.py)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    main(args.dataset, args.batch_size)
