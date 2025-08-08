import torch
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings


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

# Load Data
def load_goodscents_subset(filepath="../data/leffingwell-goodscent-merge-dataset.csv",
                           index=200,
                           shuffle=True
                           ):
    # max_rows = 500
    df = pd.read_csv(filepath)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    if index > 0:
        df = df.head(index)
    else:
        df = df.tail(-1*index)
    descriptor_cols = df.columns[2:]
    smiles_list, label_map = [], {}
    for _, row in df.iterrows():
        smiles = row["nonStereoSMILES"]
        labels = row[descriptor_cols].astype(int).tolist()
        if smiles and any(labels):
            smiles_list.append(smiles)
            label_map[smiles] = labels
    return smiles_list, label_map, list(descriptor_cols)



def sample(model, conditioner, label_vec, constrained=True, steps=1000, debug=True):
    x_t = torch.randn((10, 1))
    pos = torch.randn((10, 3))
    edge_index = torch.randint(0, 10, (2, 20))

    for t in reversed(range(1, steps + 1)):
        cond_embed = conditioner(label_vec.unsqueeze(0))
        pred_x, bond_logits = model(x_t, pos, edge_index, torch.tensor([t]), cond_embed)
        bond_logits = temperature_scaled_softmax(bond_logits, temperature=(1/t))
        x_t = x_t - pred_x * (1.0 / steps)

    x_t = x_t * 100.0
    x_t.relu_()
    atom_types = torch.clamp(x_t.round(), 1, 118).int().squeeze().tolist()
    ## Try limiting to only the molecules that the Scentience sensors can detect
    allowed_atoms = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl
    bond_logits.relu_()
    bond_preds = torch.argmax(bond_logits, dim=-1).tolist()
    if debug:
        print(f"\tcond_embed: {cond_embed}")
        print(f"\tx_t: {x_t}")
        print(f"\tprediction: {x_t}")
        print(f"\tbond logits: {bond_logits}")
        print(f"\tatoms: {atom_types}")
        print(f"\tbonds: {bond_preds}")

    mol = Chem.RWMol()
    idx_map = {}
    for i, atomic_num in enumerate(atom_types):
        if constrained and atomic_num not in allowed_atoms:
            continue
        try:
            atom = Chem.Atom(int(atomic_num))
            idx_map[i] = mol.AddAtom(atom)
        except Exception:
            continue

    if len(idx_map) < 2:
        print("Molecule too small or no valid atoms after filtering.")
        return ""

    bond_type_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }

    added = set()
    for i in range(edge_index.shape[1]):
        a = int(edge_index[0, i])
        b = int(edge_index[1, i])
        if a != b and (a, b) not in added and (b, a) not in added and a in idx_map and b in idx_map:
            try:
                bond_type = bond_type_map.get(bond_preds[i], Chem.BondType.SINGLE)
                mol.AddBond(idx_map[a], idx_map[b], bond_type)
                added.add((a, b))
            except Exception:
                continue
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        img = Draw.MolToImage(mol)
        img.show()
        print(f"Atom types: {atom_types}")
        print(f"Generated SMILES: {smiles}")
        return smiles
    except Exception as e:
        print(f"Sanitization error: {e}")
        return ""



def sample_batch(model, conditioner, label_vec, steps=1000, batch_size=4):
    mols = []
    for _ in range(batch_size):
        x_t = torch.randn((10, 1))
        pos = torch.randn((10, 3))
        edge_index = torch.randint(0, 10, (2, 20))

        for t in reversed(range(1, steps + 1)):
            cond_embed = conditioner(label_vec.unsqueeze(0))
            pred_x, bond_logits = model(x_t, pos, edge_index, torch.tensor([t]), cond_embed)
            x_t = x_t - pred_x * (1.0 / steps)

        x_t = x_t * 100.0
        x_t.relu_()
        atom_types = torch.clamp(x_t.round(), 1, 118).int().squeeze().tolist()
        allowed_atoms = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl
        bond_logits.relu_()

        mol = Chem.RWMol()
        idx_map = {}
        for i, atomic_num in enumerate(atom_types):
            if atomic_num not in allowed_atoms:
                continue
            try:
                atom = Chem.Atom(int(atomic_num))
                idx_map[i] = mol.AddAtom(atom)
            except Exception:
                continue

        if len(idx_map) < 2:
            continue

        bond_type_map = {
            0: Chem.BondType.SINGLE,
            1: Chem.BondType.DOUBLE,
            2: Chem.BondType.TRIPLE,
            3: Chem.BondType.AROMATIC
        }

        added = set()
        for i in range(edge_index.shape[1]):
            a = int(edge_index[0, i])
            b = int(edge_index[1, i])
            if a != b and (a, b) not in added and (b, a) not in added and a in idx_map and b in idx_map:
                try:
                    bond_type = bond_type_map.get(bond_preds[i], Chem.BondType.SINGLE)
                    mol.AddBond(idx_map[a], idx_map[b], bond_type)
                    added.add((a, b))
                except Exception:
                    continue

        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mols.append(mol)
        except Exception:
            continue
    return mols



# Validation
def validate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {}
    return True, {"MolWt": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol)}


# Testing
def test_models(test_model, test_conditioner):
    good_count: int = 0
    index: int = int(4983.0 * 0.2)  #take 20% of the dataset for testing
    smiles_list, label_map, label_names = load_goodscents_subset(index=index)
    dataset = []
    test_model.eval()
    test_conditioner.eval()
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g:
            g.y = torch.tensor(label_map[smi])
            dataset.append(g)

    for i in range(0, len(dataset)):
        print(f"Testing molecule {i+1}/{len(dataset)}")
        data = dataset[i]
        x_0, pos, edge_index, edge_attr, label_vec = data.x, data.pos, data.edge_index, data.edge_attr.view(-1), data.y
        new_smiles = sample(test_model, test_conditioner, label_vec=label_vec)
        print(new_smiles)
        valid, props = validate_molecule(new_smiles)
        print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")
        if new_smiles != "":
            good_count += 1

    percent_correct: float = float(good_count)  / float(len(dataset))
    print(f"Percent correct: {percent_correct}")


