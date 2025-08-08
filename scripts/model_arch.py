import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# UTILS: Molecule Processing with 3D Coordinates
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None

    conf = mol.GetConformer()
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    node_feats = []
    pos = []
    edge_index = []
    edge_attrs = []

    for atom in atoms:
        # Normalize atomic number
        node_feats.append([atom.GetAtomicNum() / 100.0])
        position = conf.GetAtomPosition(atom.GetIdx())
        pos.append([position.x, position.y, position.z])

    for bond in bonds:
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
        bond_type = bond.GetBondType()
        bond_class = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }.get(bond_type, 0)
        edge_attrs.extend([[bond_class], [bond_class]])

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.long)
    )

# EGNN Layer
class EGNNLayer(MessagePassing):
    def __init__(self, node_dim):
        super().__init__(aggr='add')
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, pos, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self.coord_updates = torch.zeros_like(pos)
        x_out, coord_out = self.propagate(edge_index, x=x, pos=pos)
        return x_out, pos + coord_out

    def message(self, x_i, x_j, pos_i, pos_j):
        edge_vec = pos_j - pos_i
        dist = ((edge_vec**2).sum(dim=-1, keepdim=True) + 1e-8).sqrt()
        h = torch.cat([x_i, x_j, dist], dim=-1)
        edge_msg = self.node_mlp(h)
        coord_update = self.coord_mlp(dist) * edge_vec
        return edge_msg, coord_update

    def message_and_aggregate(self, adj_t, x):
        raise NotImplementedError("This EGNN layer does not support sparse adjacency matrices.")

    def aggregate(self, inputs, index):
        edge_msg, coord_update = inputs
        aggr_msg = torch.zeros(index.max() + 1, edge_msg.size(-1), device=edge_msg.device).index_add_(0, index, edge_msg)
        aggr_coord = torch.zeros(index.max() + 1, coord_update.size(-1), device=coord_update.device).index_add_(0, index, coord_update)
        return aggr_msg, aggr_coord

    def update(self, aggr_out, x):
        msg, coord_update = aggr_out
        return x + msg, coord_update

# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, t):
        return self.net(t.view(-1, 1).float() / 1000)

# Olfactory Conditioning
class OlfactoryConditioner(nn.Module):
    def __init__(self, num_labels, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(num_labels, embed_dim)

    def forward(self, labels):
        return self.embedding(labels.float())

# EGNN Diffusion Model
class EGNNDiffusionModel(nn.Module):
    def __init__(self, node_dim, embed_dim):
        super().__init__()
        self.time_embed = TimeEmbedding(embed_dim)
        self.egnn1 = EGNNLayer(node_dim + embed_dim * 2)
        self.egnn2 = EGNNLayer(node_dim + embed_dim * 2)
        self.bond_predictor = nn.Sequential(
            nn.Linear((node_dim + embed_dim * 2) * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x_t, pos, edge_index, t, cond_embed):
        batch_size = x_t.size(0)
        t_embed = self.time_embed(t).expand(batch_size, -1)
        cond_embed = cond_embed.expand(batch_size, -1)
        x_input = torch.cat([x_t, cond_embed, t_embed], dim=1)
        x1, pos1 = self.egnn1(x_input, pos, edge_index)
        x2, pos2 = self.egnn2(x1, pos1, edge_index)
        edge_feats = torch.cat([x2[edge_index[0]], x2[edge_index[1]]], dim=1)
        bond_logits = self.bond_predictor(edge_feats)
        return x2[:, :x_t.shape[1]], bond_logits