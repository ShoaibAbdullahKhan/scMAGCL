"""
Utility functions for Dual Autoencoder model
"""

import torch
import numpy as np
import pandas as pd
import random
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import umap


# ============================================================================
# Seed and Reproducibility
# ============================================================================

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Data Augmentation
# ============================================================================

def create_augmented_data(x, mask_prob=0.25):
    """
    Create augmented data by randomly masking genes to zero and adding
    random Gaussian noise.

    Args:
        x: Input data [batch_size, n_genes]
        mask_prob: Probability of masking each gene
        noise_std: Standard deviation of the additive Gaussian noise
                   (set to 0 to disable the noise augmentation)

    Returns:
        x_aug: Augmented data with some genes masked to zero and Gaussian
               noise added to the surviving (unmasked) genes
    """
    mask = torch.bernoulli(torch.ones_like(x) * mask_prob).bool()
    x_aug = x.clone()
    x_aug[mask] = 0


    return x_aug


# ============================================================================
# Dataset
# ============================================================================

class scRNASeqDataset(Dataset):
    """PyTorch Dataset for scRNA-seq data"""
    
    def __init__(self, expression_matrix, labels=None):
        if isinstance(expression_matrix, pd.DataFrame):
            self.data = expression_matrix.values.astype(np.float32)
        else:
            self.data = expression_matrix.astype(np.float32)
        
        if labels is not None:
            self.labels = labels if isinstance(labels, np.ndarray) else labels.values
        else:
            self.labels = np.zeros(len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]


# ============================================================================
# Evaluation Metrics
# ============================================================================

def cluster_acc(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate_clustering(y_true, y_pred):
    """Evaluate clustering performance using multiple metrics"""
    unique_labels = np.unique(y_true)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_true_encoded = np.array([label_map[label] for label in y_true])
    
    acc = cluster_acc(y_true_encoded, y_pred)
    nmi = normalized_mutual_info_score(y_true_encoded, y_pred)
    ari = adjusted_rand_score(y_true_encoded, y_pred)
    
    return acc, nmi, ari


# ============================================================================
# Visualization
# ============================================================================

def visualize_and_evaluate(latent, labels, n_clusters, seed=42, save_dir='results_dual_ae'):
    """
    Perform clustering and UMAP visualization
    
    Args:
        latent: Latent representations [n_samples, 128]
        labels: True labels
        n_clusters: Number of clusters for K-means
        seed: Random seed
        save_dir: Directory to save results
        
    Returns:
        tuple: (accuracy, NMI, ARI)
    """
    from sklearn.cluster import KMeans
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nPerforming clustering on 128-dim latent space...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    cluster_labels = kmeans.fit_predict(latent)
    
    # Save cluster labels
    pd.DataFrame(cluster_labels, columns=['cluster_label']).to_csv(
        os.path.join(save_dir, "cluster_labels.csv"), index=False
    )
    
    # Evaluate
    acc, nmi, ari = evaluate_clustering(labels, cluster_labels)
    print(f"CA={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
    
    # UMAP visualization
    print("\nGenerating UMAP visualization...")
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed
    )
    umap_result = umap_reducer.fit_transform(latent)
    
    # Plot UMAP
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # UMAP colored by cluster
    scatter1 = axes[0].scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=cluster_labels,
        cmap='tab20',
        s=20,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.3
    )
    axes[0].set_title(f'UMAP - Predicted Clusters\nCA={acc:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].grid(alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
    
    # UMAP colored by true labels
    scatter2 = axes[1].scatter(
        umap_result[:, 0], umap_result[:, 1],
        c=labels,
        cmap='tab20',
        s=20,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.3
    )
    axes[1].set_title('UMAP - True Labels', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].grid(alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_visualization.png"), dpi=300, bbox_inches='tight')
    #plt.show()
    
    return acc, nmi, ari