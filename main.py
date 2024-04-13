from data import create_dataset, get_dataloader
from models import SimpleLSTM
from train import train_model, adversarial_training
import time
from evaluate import evaluate, evaluate_adversarial

def main():
    dataset = create_dataset()
    loader = get_dataloader(dataset)
    
    model = SimpleLSTM()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Starting regular training...")
    train_model(model, loader, device=device)

    print("Evaluating on clean data...")
    clean_accuracy = evaluate(model, loader, device)
    print(f'Clean Data Accuracy: {clean_accuracy}%')

    print("Starting adversarial training...")
    adversarial_training(model, loader, device=device)

    print("Evaluating on adversarial data...")
    adv_accuracy = evaluate(model, loader, device)
    print(f'Adversarial Data Accuracy: {adv_accuracy}%')

if __name__ == '__main__':
    main()


# Initialize model and optimizer
model_pgd = SimpleLSTM().to(device)
optimizer_pgd = optim.Adam(model_pgd.parameters(), lr=0.01)

model_trades = SimpleLSTM().to(device)
optimizer_trades = optim.Adam(model_trades.parameters(), lr=0.01)

# PGD Training
print("Starting PGD adversarial training...")
train_pgd(model_pgd, loader, optimizer_pgd, epochs=5, device=device)

# TRADES Training
print("Starting TRADES adversarial training...")
train_trades(model_trades, loader, optimizer_trades, epochs=5, device=device)



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
