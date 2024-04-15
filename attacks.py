import torch
import numpy as np
import torch.fft

def simba_single(model, x, y, num_iters=1000, epsilon=0.1, targeted=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.clone().detach().to(device)  # Ensure that x is a copy and moved to the correct device
    y = y.to(device)
    batch_size, seq_len, num_features = x.shape
    n_dims = seq_len * num_features

    perm = torch.randperm(n_dims).to(device)
    last_prob = get_probs(model, x, y)
    queries = 0
    l2_norms = []

    for i in range(num_iters):
        diff = torch.zeros(batch_size, seq_len, num_features).to(device)
        index = perm[i % n_dims]
        feature_index = index % num_features
        time_step_index = index // num_features
        diff[:, time_step_index, feature_index] = epsilon

        for sign in [-1, 1]:
            perturbed_x = (x + sign * diff).clamp(0, 1)
            perturbed_prob = get_probs(model, perturbed_x, y)
            queries += 1

            # Element-wise comparison to decide which samples to update
            if targeted:
                improved = perturbed_prob > last_prob
            else:
                improved = perturbed_prob < last_prob

            # Update x where improvements have occurred
            x[improved] = perturbed_x[improved]
            last_prob[improved] = perturbed_prob[improved]

        l2_norms.append(torch.norm(sign * diff).item())

        if i % 100 == 0:
            print(f'Iteration {i}: Probability = {last_prob.mean().item()}')

    return x, queries, torch.tensor(l2_norms).mean()


def simba_temporal(model, x, y, num_iters=1000, epsilon=0.1, targeted=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.clone().detach().to(device)  # Ensure that x is a copy and moved to the correct device
    y = y.to(device)
    seq_len = x.size(1)  # Assuming x is [batch, sequence_length, features]

    last_prob = get_probs(model, x, y)
    for t in range(seq_len):  # Sequential perturbation
        for i in range(num_iters):
            diff = torch.zeros_like(x)
            diff[:, t, :] = epsilon  # Perturb current time step across all features

            # Apply perturbation in both directions and check model output
            perturbed_x = (x + diff).clamp(0, 1)
            perturbed_prob = get_probs(model, perturbed_x, y)
            if (targeted and perturbed_prob > last_prob) or (not targeted and perturbed_prob < last_prob):
                x = perturbed_x
                last_prob = perturbed_prob
            else:
                perturbed_x = (x - diff).clamp(0, 1)
                perturbed_prob = get_probs(model, perturbed_x, y)
                if (targeted and perturbed_prob > last_prob) or (not targeted and perturbed_prob < last_prob):
                    x = perturbed_x
                    last_prob = perturbed_prob

        if t % 10 == 0:
            print(f'Time Step {t}: Last probability = {last_prob.mean().item()}')

    return x



def simba_frequency(model, x, y, num_iters=100, epsilon=0.05, targeted=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.clone().detach().to(device)  # Ensure that x is a copy and moved to the correct device
    y = y.to(device)
    seq_len = x.size(1)  # Assuming x is [batch, sequence_length, features]

    last_prob = get_probs(model, x, y)
    for i in range(num_iters):
        # Transform to frequency domain
        freqs = torch.fft.rfft(x, dim=1)

        # Create perturbation in the frequency domain
        perturbation = torch.zeros_like(freqs)
        random_idx = np.random.randint(freqs.shape[1])
        perturbation[:, random_idx, :] += epsilon

        # Apply perturbation and transform back
        perturbed_freqs = freqs + perturbation
        perturbed_x = torch.fft.irfft(perturbed_freqs, n=seq_len, dim=1).clamp(0, 1)

        perturbed_prob = get_probs(model, perturbed_x, y)
        if (targeted and perturbed_prob > last_prob) or (not targeted and perturbed_prob < last_prob):
            x = perturbed_x
            last_prob = perturbed_prob

    return x
