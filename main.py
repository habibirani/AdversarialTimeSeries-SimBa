from data import get_dataset, get_dataloader
from src import attacks, evaluate, models,train
from src.attacks import simba_single, simba_frequency, simba_temporal
from src.evaluate import evaluate
from src.models import CNN1D, LSTMModel, TransformerEncoder
from src.train import train_model, train_pgd, adversarial_training, train_trades

import time
import torch 
import torch.nn as nn
import torch.optim as optim

def main():
    ds_list = ["UniMiB SHAR", "UCI HAR", "TWristAR", "Leotta_2021", "Gesture Phase Segmentation"]
    dataset = get_dataset(ds_list)
    loader = get_dataloader(dataset)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize models and optimizers for different training procedures
    model_pgd = CNN1D().to(device)  # Example model
    optimizer_pgd = optim.Adam(model_pgd.parameters(), lr=0.01)

    model_trades = LSTMModel().to(device)  # Example model
    optimizer_trades = optim.Adam(model_trades.parameters(), lr=0.01)

    model_yours = TransformerEncoder().to(device)  # Example model
    optimizer_yours = optim.Adam(model_yours.parameters(), lr=0.01)

    # Regular Training
    print("Starting regular training...")
    train_model(model_yours, loader, device=device)

    # Evaluate on clean data
    print("Evaluating on clean data...")
    clean_accuracy = evaluate(model_yours, loader, device)
    print(f'Clean Data Accuracy: {clean_accuracy}%')

    # Adversarial Training
    print("Starting adversarial training...")
    adversarial_training(model_yours, loader, device=device)

    # Evaluate on adversarial data
    print("Evaluating on adversarial data...")
    adv_accuracy = evaluate(model_yours, loader, device)
    print(f'Adversarial Data Accuracy: {adv_accuracy}%')

    # PGD and TRADES Training
    print("Starting PGD adversarial training...")
    train_pgd(model_pgd, loader, optimizer_pgd, epochs=5, device=device)

    print("Starting TRADES adversarial training...")
    train_trades(model_trades, loader, optimizer_trades, epochs=5, device=device)

    # Full Evaluation
    def full_evaluation(model, loader, device):
        start_time = time.time()
        clean_accuracy = evaluate(model, loader, device)
        adv_accuracy, avg_perturbation = evaluate_adversarial(model, loader, device)
        end_time = time.time()
        
        training_time = end_time - start_time
        return clean_accuracy, adv_accuracy, avg_perturbation, training_time

    # Assuming models are already trained
    clean_acc_pgd, adv_acc_pgd, perturb_pgd, time_pgd = full_evaluation(model_pgd, loader, device)
    clean_acc_trades, adv_acc_trades, perturb_trades, time_trades = full_evaluation(model_trades, loader, device)
    clean_acc_ours, adv_acc_ours, perturb_ours, time_ours = full_evaluation(model_yours, loader, device)

    # Print or log results for comparison
    print(f"PGD Training - Clean Acc: {clean_acc_pgd}, Adv Acc: {adv_acc_pgd}, Perturb: {perturb_pgd}, Time: {time_pgd}")
    print(f"TRADES Training - Clean Acc: {clean_acc_trades}, Adv Acc: {adv_acc_trades}, Perturb: {perturb_trades}, Time: {time_trades}")
    print(f"Our Method - Clean Acc: {clean_acc_ours}, Adv Acc: {adv_acc_ours}, Perturb: {perturb_ours}, Time: {time_ours}")

if __name__ == '__main__':
    main()

