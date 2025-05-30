import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchtext.vocab import GloVe


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)  # +1 for the pseudo-label
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        h = F.relu(self.fc1(torch.cat((x, y.unsqueeze(1)), dim=1)))  # concatenate pseudo-label
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_out(h)

class GenderClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(GenderClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, 1)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.gender_classifier = GenderClassifier(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        gender_pred = self.gender_classifier(z)
        return reconstruction, gender_pred, mu, logvar, z

    def q_z_given_xy(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


def smooth_abs(x, epsilon=1e-8):
    return torch.sqrt(x**2 + epsilon)


class HistogramApproximator:
    def __init__(self, num_bins=100, range_min=-5, range_max=5):
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.histograms = None

    def update(self, z, y):
        if self.histograms is None:
            self.histograms = [torch.zeros((2, self.num_bins)) for _ in range(z.shape[1])]  # 2 bins for y = 0 and y = 1

        for i in range(z.shape[1]):
            for j in range(2):
                hist, _ = torch.histogram(z[y == j, i], bins=self.num_bins, range=(self.range_min, self.range_max))
                self.histograms[i][j] += hist

    def get_probabilities(self, z, y):
        probs = torch.ones(z.shape[0], device=z.device)
        for i in range(z.shape[1]):
            for j in range(2):
                indices = torch.clamp(((z[y == j, i] - self.range_min) / (self.range_max - self.range_min) * self.num_bins).long(), 0, self.num_bins - 1)
                probs[y == j] *= self.histograms[i][j][indices] + 1e-10  # Add small constant to avoid division by zero
        return probs


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def debias_embeddings(emb, V_f, V_m, V_s, V_n, alpha, beta, eta, T, l, lambda_rec, lambda_g, lambda_KL, batch_size, gamma, lambda_debias):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = emb.vectors.shape[1]
    hidden_dim = l
    latent_dim = l
    output_dim = input_dim

    vae = VAE(input_dim, hidden_dim, latent_dim, output_dim).to(device)
    optimizer = Adam(vae.parameters(), lr=eta)
    E = emb.vectors.clone().detach().to(device)

    # Create pseudo-labels
    pseudo_labels = torch.zeros(len(emb.itos), device=device)
    for word in V_m:
        if word in emb.stoi:
            pseudo_labels[emb.stoi[word]] = 1
    for word in V_f:
        if word in emb.stoi:
            pseudo_labels[emb.stoi[word]] = 0

    gender_pairs = list(zip(V_f, V_m))
    histogram_approximator = HistogramApproximator()

    # Initial histogram approximation
    with torch.no_grad():
        _, _, _, _, z = vae(E, pseudo_labels)
        histogram_approximator.update(z.cpu(), pseudo_labels.cpu())

    for t in range(1, T + 1):
        # Reweighting step
        with torch.no_grad():
            _, _, _, _, z = vae(E, pseudo_labels)
            probs = histogram_approximator.get_probabilities(z, pseudo_labels)
            probs = torch.clamp(probs, min=1e-10)
            probs = probs / probs.sum()
            weights = 1 / (probs + alpha)
            weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
            weights = weights.clamp(min=0)
            weights = weights / weights.sum()
            reweighted_z = z * weights.unsqueeze(1)
            E_reweighted = vae.decoder(reweighted_z)

        # Calculate mean vectors for female and male words using reweighted embeddings
        v_f = torch.mean(torch.stack([E_reweighted[emb.stoi[word]] for word in V_f if word in emb.stoi]), dim=0)
        v_m = torch.mean(torch.stack([E_reweighted[emb.stoi[word]] for word in V_m if word in emb.stoi]), dim=0)

        # Training step
        indices = torch.randperm(len(E_reweighted))[:batch_size]
        E_batch = E_reweighted[indices]
        y_batch = pseudo_labels[indices]

        reconstruction, gender_pred, mu, logvar, z = vae(E_batch, y_batch)

        recon_loss = F.mse_loss(reconstruction, E_batch)
        kl_loss = kl_divergence(mu, logvar)
        L_g1 = F.binary_cross_entropy(gender_pred.squeeze(), y_batch)

        L_b = torch.tensor(0.0, device=device)
        for w_f, w_m in gender_pairs:
            if emb.stoi.get(w_f) is not None and emb.stoi.get(w_m) is not None:
                z_f, _, _ = vae.q_z_given_xy(E_reweighted[emb.stoi[w_f]].unsqueeze(0), torch.tensor([0], device=device))
                z_m, _, _ = vae.q_z_given_xy(E_reweighted[emb.stoi[w_m]].unsqueeze(0), torch.tensor([1], device=device))

                gender_pred_f = vae.gender_classifier(z_f)
                gender_pred_m = vae.gender_classifier(z_m)

                L_b += (gender_pred_f.mean() - gender_pred_m.mean()).pow(2)

        L_b = L_b / len(gender_pairs) if gender_pairs else torch.tensor(0.0, device=device)
        L_g = L_g1 + gamma * L_b

        L_debias = torch.tensor(0.0, device=device)
        for w in V_s:
            if w in emb.stoi and emb.stoi[w] in indices:
                e_w = reconstruction[indices == emb.stoi[w]]
                if e_w.shape[0] > 0:
                    s_f = cosine_similarity(e_w.mean(dim=0), v_f)
                    s_m = cosine_similarity(e_w.mean(dim=0), v_m)
                    L_debias += smooth_abs(torch.tensor(s_f, device=device) - torch.tensor(s_m, device=device))
        L_debias = L_debias / len(V_s) if V_s else torch.tensor(0.0, device=device)

        total_loss = lambda_rec * recon_loss + lambda_KL * kl_loss + lambda_g * L_g + lambda_debias * L_debias

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update histogram approximator
        with torch.no_grad():
            _, _, _, _, z = vae(E_reweighted, pseudo_labels)
            histogram_approximator.update(z.cpu(), pseudo_labels.cpu())

        if t % 100 == 0:
            print(f"Epoch {t}: Loss = {total_loss.item()}, L_g1 = {L_g1.item()}, L_b = {L_b.item()}, L_debias = {L_debias.item()}")

    # Use the final reweighted embeddings
    emb_hat = {word: E_reweighted[i].cpu().numpy() for i, word in enumerate(emb.itos)}
    return emb_hat


glove_embeddings = GloVe(name='840B', dim=300) 


emb_hat = debias_embeddings(glove_embeddings, V_f, V_m, V_s, V_n, alpha, beta, eta, T, l, lambda_rec, lambda_g, lambda_KL, batch_size, gamma, lambda_debias)

# Save debiased embeddings
with open('/content/drive/My Drive/D-VAN-glove-6B.txt', 'w', encoding='utf-8') as f:
    for word, embedding in emb_hat.items():
        f.write(f"{word} {' '.join(map(str, embedding))}\n")
