import torch

from .model_arch import EGNNDiffusionModel, OlfactoryConditioner
from .utils import load_goodscents_subset, validate_molecule, sample, sample_batch, smiles_to_graph
from .train import train



# -------- Main --------
if __name__ == '__main__':
    # Batch data if desired
    SHOULD_BATCH: bool = False

    # Load the dataset
    smiles_list, label_map, label_names = load_goodscents_subset(
        filepath="../data/leffingwell-goodscent-merge-dataset.csv",
        index=500,
        shuffle=True)
    num_labels = len(label_names)
    dataset = []

    # Convert all SMILES to graphs for the dataset
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g:
            g.y = torch.tensor(label_map[smi])
            dataset.append(g)

    # Init the models
    model = EGNNDiffusionModel(node_dim=1, embed_dim=8)
    conditioner = OlfactoryConditioner(num_labels=num_labels, embed_dim=8)

    # Begin training
    train(model, conditioner, dataset, epochs=500)

    # Test trained model
    test_label_vec = torch.zeros(num_labels)
    if "floral" in label_names:
        test_label_vec[label_names.index("floral")] = 0
    if "fruity" in label_names:
        test_label_vec[label_names.index("fruity")] = 1
    if "musky" in label_names:
        test_label_vec[label_names.index("musky")] = 0

    model.eval()
    conditioner.eval()

    if SHOULD_BATCH:
        new_smiles_list = sample_batch(model, conditioner, label_vec=test_label_vec)
        for new_smiles in new_smiles_list:
            print(new_smiles)
            valid, props = validate_molecule(new_smiles)
            print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")
    else:
        new_smiles = sample(model, conditioner, label_vec=test_label_vec)
        print(new_smiles)
        valid, props = validate_molecule(new_smiles)
        print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")