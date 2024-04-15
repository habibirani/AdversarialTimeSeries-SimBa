import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam
from utils import simba_single
from advertorch.attacks import LinfPGDAttack
from trades import trades_loss



def train_model(model, data_loader, epochs=5, lr=0.01, device='cpu'):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}: Training Loss: {loss.item()}')

def adversarial_training(model, data_loader, epochs=5, adv_ratio=0.5, lr=0.01, device='cpu'):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            # Regular training
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Adversarial training
            if torch.rand(1).item() < adv_ratio:
                adv_data, _, _ = simba_single(model, data, labels)
                adv_outputs = model(adv_data)
                adv_loss = criterion(adv_outputs, labels)
                optimizer.zero_grad()
                adv_loss.backward()
                optimizer.step()

        print(f'Epoch {epoch+1}: Adv Training Loss: {adv_loss.item() if "adv_loss" in locals() else "N/A"}')


def train_pgd(model, data_loader, optimizer, epochs, device, epsilon=0.3, step_size=0.01, num_steps=40):
    pgd_criterion = torch.nn.CrossEntropyLoss()
    pgd_attacker = LinfPGDAttack(
        model, loss_fn=pgd_criterion, eps=epsilon,
        nb_iter=num_steps, eps_iter=step_size, rand_init=True, clip_min=-1.0, clip_max=1.0,
        targeted=False)

    model.train()
    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # Generate adversarial data
            adv_data = pgd_attacker.perturb(data, target)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = pgd_criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")    


def train_trades(model, data_loader, optimizer, epochs, device, beta=6.0, step_size=0.007, epsilon=0.031, num_steps=10):
    model.train()
    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # Calculate loss
            loss = trades_loss(model=model, x_natural=data, y=target, optimizer=optimizer,
                               step_size=step_size, epsilon=epsilon, perturb_steps=num_steps, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
