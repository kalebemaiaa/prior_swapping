from torch import autograd, nn
from typing import Callable

import torch

class Otimizador(nn.Module):
    def __init__(self, use_hutchinson: bool = True, num_probes: int = 16, device = None, dtype=torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        # FIXME: resolver esses 2 ultimos argumntos -> para que servem, utilidade, etc...
        self.use_hutchinson = use_hutchinson
        self.num_probes = num_probes

    def laplaciano(self, theta: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if grad.dim() == 1:
            grad = grad.unsqueeze(1)
        batch_size, d = theta.shape
        traces = torch.zeros(batch_size, device=self.__device, dtype=self.__dtype)

        if self.use_hutchinson:
            for _ in range(self.num_probes):
                v= torch.randint(0, 2, theta.shape, device=self.__device, dtype=self.__dtype) * 2 - 1
                g_v = (grad * v).sum()
                Hv = autograd.grad(g_v, theta, create_graph=True, retain_graph=True)[0]
                traces += (v * Hv).sum(dim=-1)
            return traces / float(self.num_probes)

        for i in range(d):
            g_i = grad[:, i].sum()
            # d^2logp/dthete_i^2
            second = autograd.grad(g_i, theta, create_graph=True, retain_graph=True)[0]
            traces += second[:, i]
        return traces

    def score_matching_loss(self, log_density_SM: Callable, alpha: torch.Tensor, thetas: torch.Tensor, N_dataset: int, K_alpha_samples: int) -> torch.Tensor:
        thetas = thetas.clone().requires_grad_(True)
        logp = log_density_SM(thetas, alpha, N_dataset, K_alpha_samples)
        if torch.isnan(logp).any() or torch.isinf(logp).any():
            print("LOGP INVALIDO")
            print(logp)
            raise Exception("logp explodiu")

        gradiente = autograd.grad(logp.sum(), thetas, create_graph=True)[0]
        laplaciano = self.laplaciano(thetas, gradiente)
        loss = 0.5 * (gradiente ** 2).sum(dim=-1) + laplaciano
        return loss.mean()

    def find_alpha_star(self, log_density_SM: Callable, samples: torch.Tensor, N_dataset: int, n_steps: int, K_alpha_samples: int, alpha_dim: int, learning_rate: float = 1e-3, batch_size: int = 60, verbose = True, init_form_samples: bool = False):
        Tf = samples.shape[0]

        # alpha_init = torch.zeros(K_alpha_samples, alpha_dim).to(DEVICE)
        alpha_init = 0.1 * torch.randn(K_alpha_samples, alpha_dim, device=self.__device)
        # alpha_init = torch.zeros(K_alpha_samples, device=self.__device, dtype=self.__dtypes)
        alphas = nn.Parameter(alpha_init)

        optimizer = torch.optim.Adam([alphas], lr=learning_rate)
        samples = samples.to(self.__device)

        self.history = {
            "loss": [],
            "alpha": []
        }

        print("Começando descida...")
        for _ in range(n_steps):
            optimizer.zero_grad()

            # df = pd.DataFrame(batch_samples.cpu().numpy())
            # df.to_csv(f"./lixo_{_}.csv", index=False)
            loss = self.score_matching_loss(log_density_SM, alphas, samples, N_dataset, K_alpha_samples)
            # loss = loss * (batch_samples.shape[0] / N_dataset)
            loss.backward()
            nn.utils.clip_grad_norm_([alphas], 10.0)
            optimizer.step()

            self.history["loss"].append(loss.item())
            self.history["alpha"].append(alphas.detach().clone())

            if verbose and (_%100 == 0 or _ == n_steps - 1):
                print(f"[{_:04d}] Loss = {loss.item():.4f}")

        return alphas.detach()