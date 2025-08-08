import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt


# -------- Noise and Training --------
def add_noise(x_0, noise, t):
    return x_0 + noise * (t / 1000.0)


def plot_data(mu, sigma, color, title):
    all_losses = np.array(mu)
    sigma_losses = np.array(sigma)
    x = np.arange(len(mu))
    plt.plot(x, all_losses, f'{color}-')
    plt.fill_between(x, all_losses - sigma_losses, all_losses + sigma_losses, color=color, alpha=0.2)
    plt.legend(['Mean Loss', 'Variance of Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


def train(model, conditioner, dataset, epochs=10):
    model.train()
    conditioner.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(conditioner.parameters()), lr=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    all_bond_losses: list = []
    all_noise_losses: list = []
    all_losses: list = []
    all_sigma_bond_losses: list = []
    all_sigma_noise_losses: list = []
    all_sigma_losses: list = []

    for epoch in range(epochs):
        total_bond_loss = 0
        total_noise_loss = 0
        total_loss = 0
        sigma_bond_losses: list = []
        sigma_noise_losses: list = []
        sigma_losses: list = []

        for data in dataset:
            x_0, pos, edge_index, edge_attr, labels = data.x, data.pos, data.edge_index, data.edge_attr.view(-1), data.y
            if torch.any(edge_attr >= 4) or torch.any(edge_attr < 0) or torch.any(torch.isnan(x_0)):
                continue  # skip corrupted data
            t = torch.tensor([random.randint(1, 1000)])
            noise = torch.randn_like(x_0)
            x_t = add_noise(x_0, noise, t)
            cond_embed = conditioner(labels)
            pred_noise, bond_logits = model(x_t, pos, edge_index, t, cond_embed)
            loss_noise = F.mse_loss(pred_noise, noise)
            loss_bond = ce_loss(bond_logits, edge_attr)
            loss = loss_noise + loss_bond
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_bond_loss += loss_bond.item()
            total_noise_loss += loss_noise.item()
            total_loss += loss.item()
            sigma_bond_losses.append(loss_bond.item())
            sigma_noise_losses.append(loss_noise.item())
            sigma_losses.append(loss.item())

        all_bond_losses.append(total_bond_loss)
        all_noise_losses.append(total_noise_loss)
        all_losses.append(total_loss)
        all_sigma_bond_losses.append(torch.std(torch.tensor(sigma_bond_losses)))
        all_sigma_noise_losses.append(torch.std(torch.tensor(sigma_noise_losses)))
        all_sigma_losses.append(torch.std(torch.tensor(sigma_losses)))
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Noise Loss = {total_noise_loss:.4f}, Bond Loss = {total_bond_loss:.4f}")

    plot_data(mu=all_bond_losses, sigma=all_sigma_bond_losses, color='b', title="Bond Loss")
    plot_data(mu=all_noise_losses, sigma=all_sigma_noise_losses, color='r', title="Noise Loss")
    plot_data(mu=all_losses, sigma=all_sigma_losses, color='g', title="Total Loss")

    plt.plot(all_bond_losses)
    plt.plot(all_noise_losses)
    plt.plot(all_losses)
    plt.legend(['Bond Loss', 'Noise Loss', 'Total Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()
    return model, conditioner


# Generation
def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)
