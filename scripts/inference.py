import torch
import json

from .model_arch import EGNNDiffusionModel, OlfactoryConditioner
from .utils import load_goodscents_subset, sample, validate_molecule


# Get the data
smiles_list, label_map, label_names = load_goodscents_subset(index=500)
num_labels = len(label_names)

# Load the models
model = EGNNDiffusionModel(node_dim=1, embed_dim=8)
model.load_state_dict(torch.load('/models/constrained/egnn_state_dict.pth'))
model.eval() # Set to evaluation mode if you are not training

conditioner = OlfactoryConditioner(num_labels=num_labels, embed_dim=8)
conditioner.load_state_dict(torch.load('/models/constrained/olfactory_conditioner_state_dict.pth'))
conditioner.eval() # Set to evaluation mode if you are not training


# Build descriptor and aroma lists from dataset
descriptor_list: list = []
aroma_vec_list: list = []
with open('data/olfaction-vision-language-dataset.json', 'r') as file:
    json_string = json.load(file)
    data = json.loads(json_string)
    for item in data:
        item_dict: dict = dict(item)
        if "descriptors" in item_dict.keys():
            descriptor_list.append(item_dict["descriptors"])
        if "aroma_vec" in item_dict.keys():
            aroma_vec_list.append(item_dict["aroma_vec"])


# Begin testing on goodscents dataset
smiles_list, label_map, label_names = load_goodscents_subset(index=1000)
num_labels = len(label_names)
count: int = 0
for i in range(0, len(descriptor_list)):
    test_label_vec = torch.zeros(num_labels)
    for descriptor in descriptor_list[i]:
        if descriptor in label_names:
            test_label_vec[label_names.index(descriptor)] = 1

    # Get the SMILES string for each sample
    new_smiles = sample(model, conditioner, label_vec=test_label_vec)
    print(new_smiles)

    # Validate the molecular propertires
    valid, props = validate_molecule(new_smiles)
    print(f"Generated SMILES: {new_smiles}\nValid: {valid}, Properties: {props}")
    if new_smiles != "":
        count += 1

    # Test accuracy
    percent_correct: float = float(count)  / float(len(aroma_vec_list)) * 100.0
    print(f"Percent correct: {percent_correct}")
