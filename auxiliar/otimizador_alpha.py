import torch

class Otimizador:
    def __init__(self):
        pass

    def score_matching_loss(self, alpha, thetas, prior_false, likelihood, n, k):
        def log_p_tilde(theta, alpha, prior_false, likelihood, n, k):
            val = torch.log(prior_false(theta))
            for j in range(k):
                val = val + (n / k) * torch.log(likelihood(alpha[j], theta))
            return val

        losses = []
        for theta in thetas:
            theta = theta.clone().detach().requires_grad_(True)
            logp = log_p_tilde(theta, alpha, prior_false, likelihood, n, k)

            # gradiente da log-densidade
            grad = torch.autograd.grad(logp, theta, create_graph=True)[0]

            # Laplaciano
            laplace = sum(torch.autograd.grad(grad[d], theta, retain_graph=True)[0][d] for d in range(theta.shape[0]))

            # loss de score matching
            losses.append(0.5 * (grad**2).sum() + laplace)

        return torch.mean(torch.stack(losses))

    def find_alpha_star(self, samples, alpha_dimension: int, false_prior, likelihood_function, num_amostras_dados_X: int, n_steps: int = 1000, k_blocos: int = 10):
        alpha = torch.randn(alpha_dimension, requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=0.01)

        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = self.score_matching_loss(alpha, samples, false_prior, likelihood_function, num_amostras_dados_X, k_blocos)
            loss.backward()
            optimizer.step()

        return alpha