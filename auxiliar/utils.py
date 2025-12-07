from typing import Callable
import torch

def f_next_state(theta_curr, proposal_std=1.0, **kwargs):
    noise = torch.normal(mean=torch.zeros_like(theta_curr), std=proposal_std)
    theta_prop = theta_curr + noise
    return theta_prop


def make_log_density_SM(log_fake_prior: Callable, log_likelihood: Callable):
    def log_density_score_matching(alphas, theta, n, k):
        # alphas: shape (k, p)
        # theta: shape (p,)
        
        # prior part
        lp = log_fake_prior(theta)               # scalar

        # likelihood part (vectorized)
        # log_likelihood returns tensor of shape (k,)
        ll = log_likelihood(alphas, theta)       # vector

        # weighted sum according to (n/k)
        ll_sum = ll.sum() * (n / k)

        return lp + ll_sum

    return log_density_score_matching