from typing import Callable

import torch

def f_next_state(theta_curr, proposal_std=1.0, **kwargs):
    noise = torch.normal(mean=torch.zeros_like(theta_curr), std=proposal_std)
    theta_prop = theta_curr + noise
    return theta_prop


def make_f_verossimilhanca(f_posteriori_false: Callable, f_priori_alvo: Callable, f_priori_false: Callable) -> Callable:
    def f_verossimilhanca(theta):
        return f_posteriori_false(theta) * f_priori_alvo(theta) / f_priori_false(theta)

    return f_verossimilhanca

def get_weights_IS(amostras, false_log_priori, target_log_priori):
    # h(\theta) * (target_log(\theta) - false_log(\theta)
    log_w = torch.stack([target_log_priori(theta) - false_log_priori(theta) for theta in amostras])
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)
    w = w / w.sum()
    return w

# @title Normal exact posteriori
def sample_false_posterior_linear_regression(
    X, y, numero_samples=1000, tau=1.0, sigma2=None, add_intercept=True, dtype=torch.float64, device='cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    X = torch.as_tensor(X, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device).reshape(-1, 1)

    n, d = X.shape

    # como não temos sigma, vamos estimar usando o conjunto de dados
    if sigma2 is None:
        # theta_ols = (X^T X)^{-1} X^T y
        XtX = X.T @ X
        XtX_inv = torch.linalg.inv(XtX)
        theta_ols = XtX_inv @ (X.T @ y)
        resid = y - X @ theta_ols
        sigma2 = (resid.T @ resid) / (n - d)
        sigma2 = float(sigma2.squeeze())

    tau2 = float(tau**2)

    Sigma_inv = (X.T @ X) / sigma2 + torch.eye(d, dtype=dtype, device=device) / tau2
    Sigma_post = torch.linalg.inv(Sigma_inv)
    mu_post = (Sigma_post @ (X.T @ y) / sigma2).squeeze()

    L = torch.linalg.cholesky(Sigma_post)
    mvn = torch.distributions.MultivariateNormal(loc=mu_post, scale_tril=L)
    samples = mvn.sample((numero_samples,))

    return samples, mu_post, Sigma_post, sigma2