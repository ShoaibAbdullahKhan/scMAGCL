
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
mse_criterion = nn.MSELoss()

class Autoencoder(nn.Module):
    
    def __init__(self, input_dim, dropout=0.1):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder1 = nn.Linear(input_dim, 512)
        self.encoder2 = nn.Linear(512, 128)  # Latent spaces
        
        # Decoder layers
        self.decoder1 = nn.Linear(128, 512)
        self.decoder2 = nn.Linear(512, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):

        # Encoder
        h1 = self.relu(self.encoder1(x))  # [batch, 1024]
        #h1 = self.dropout(h1)
        
        #h2 = self.relu(self.encoder2(h1))  # [batch, 512]
        #h2 = self.dropout(h2)
        
        h3 = self.relu(self.encoder2(h1))  # [batch, 128] - Latent
        
        # Decoder
        d1 = self.relu(self.decoder1(h3))  # [batch, 512]
        #d1 = self.dropout(d1)
        
        #d2 = self.relu(self.decoder2(d1))  # [batch, 1024]
        #d2 = self.dropout(d2)
        
        recon = self.decoder2(d1)  # [batch, input_dim]
        
        return {
            'input' : x,
            'h1_512': h1,
            'h2_128': h3,  
            'd1_512': d1,
            'recon': recon
        }

class MultiHeadAttentionGraph(nn.Module):

    
    def __init__(self, dim, num_heads=8):
        super(MultiHeadAttentionGraph, self).__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Linear transformations
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_o = nn.Linear(num_heads, 1)
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x1, x2, cross=False):

        batch_size1 = x1.shape[0]
        batch_size2 = x2.shape[0]

        # Self-attention: Q and K from same representation
        Q = self.W_q(x1)  # [batch, dim]
        K = self.W_k(x2)  # [batch, dim]
        
        # Reshape for multi-head
        Q = Q.view(batch_size1, self.num_heads, self.head_dim).transpose(0, 1)  # [heads, batch, head_dim]
        K = K.view(batch_size2, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [heads, batch, batch]
        attn_weights = F.softmax(scores, dim=-1)
        
        # sigmoid keeps graph values in (0,1) — negative raw outputs cause NaN in cl_loss
        graph = torch.sigmoid(self.W_o(attn_weights.permute(1, 2, 0)).squeeze(-1))  # [batch, batch]

        return graph

def sim(z1, z2, hidden_norm):
    if hidden_norm:
        # eps avoids NaN from zero-norm vectors (dead ReLU neurons)
        z1 = F.normalize(z1, dim=1, eps=1e-8)
        z2 = F.normalize(z2, dim=1, eps=1e-8)
    return torch.mm(z1, z2.T)


def cl_loss(z, z_aug, adj,adj_combine, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive =  (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj_combine)).sum(1)
    denominator = intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag()

    loss = torch.log((positive / denominator).clamp(min=1e-8))

    adj_count = (torch.sum(adj, 1) * 2 + 1).clamp(min=1e-8)
    loss = loss / adj_count

    return -torch.mean(loss, 0)


def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, adj_combine,adj_aug_combine ,tau, hidden_norm=True):


    loss = alpha1 * cl_loss(z, z_aug, adj,adj_combine, tau, hidden_norm) \
        + alpha2 * cl_loss(z_aug, z, adj_aug,adj_aug_combine, tau, hidden_norm) \
        +mse_criterion(z, z_aug) \

    return loss



class scMAGCL(nn.Module):
    @staticmethod
    def _valid_heads(dim, num_heads):
        h = num_heads
        while h > 1 and dim % h != 0:
            h -= 1
        return h
    def __init__(self, input_dim, num_heads=8, dropout=0.1, phi=.6, k=15):
        super(scMAGCL, self).__init__()
        self.input_dim = input_dim
        recon_heads = self._valid_heads(input_dim, 40)

        # Two autoencoders
        self.ae1 = Autoencoder(input_dim, dropout)

        self.attn_ae1_128 = MultiHeadAttentionGraph(128, 8)
        self.attn_ae1_512 = MultiHeadAttentionGraph(512, 8)
        self.attn_ae1_recon = MultiHeadAttentionGraph(input_dim, recon_heads)

        
        # Graph fusion networks
        
    def forward(self, x, x_aug):

        # Forward through both autoencoders
        ae1_out = self.ae1(x)
        ae2_out = self.ae1(x_aug)
        

        graph_ae1_128 = self.attn_ae1_128(ae1_out['h2_128'], ae1_out['h2_128'])
        graph_ae1_d_512 = self.attn_ae1_512(ae1_out['d1_512'], ae1_out['d1_512'])
        graph_ae1_recon = self.attn_ae1_recon(ae1_out['recon'], ae1_out['recon'])
        #print(graph_ae1_128, graph_ae1_d_512, graph_ae1_recon)
        graphs_ae1 = [graph_ae1_128, graph_ae1_d_512, graph_ae1_recon]


        graph_ae2_128 = self.attn_ae1_128(ae2_out['h2_128'], ae2_out['h2_128'])
        graph_ae2_d_512 = self.attn_ae1_512(ae2_out['d1_512'], ae2_out['d1_512'])
        graph_ae2_recon = self.attn_ae1_recon(ae2_out['recon'], ae2_out['recon'])
        
        graphs_ae2 = [graph_ae2_128,graph_ae2_d_512, graph_ae2_recon]
        

        graph_ae12_128 = self.attn_ae1_128(ae1_out['h2_128'], ae2_out['h2_128'])
        graph_ae12_d_512 = self.attn_ae1_512(ae1_out['d1_512'], ae2_out['d1_512'])
        graph_ae12_recon = self.attn_ae1_recon(ae1_out['recon'], ae2_out['recon'])
        
        graphs_ae12 = [graph_ae12_128,graph_ae12_d_512, graph_ae12_recon]
        
        graph_ae21_128 = self.attn_ae1_128(ae2_out['h2_128'], ae1_out['h2_128'])
        graph_ae21_d_512 = self.attn_ae1_512(ae2_out['d1_512'], ae1_out['d1_512'])
        graph_ae21_recon = self.attn_ae1_recon(ae2_out['recon'], ae1_out['recon'])
        
        graphs_ae21 = [graph_ae21_128,graph_ae21_d_512, graph_ae21_recon]

        return {
            'ae1_out': ae1_out,
            'ae2_out': ae2_out,
            'graphs_ae1': graphs_ae1,
            'graphs_ae2': graphs_ae2,
            'graphs_ae12': graphs_ae12,
            'graphs_ae21': graphs_ae21,
        }