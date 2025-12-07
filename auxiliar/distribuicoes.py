import numpy as np
import torch


class Laplace:
    def __init__(self, media: float = 10, beta: float = 1, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.__media = torch.tensor(media, dtype=self.__dtype, device=self.__device)
        self.__beta = torch.tensor(beta, dtype=self.__dtype, device=self.__device)

    def pdf(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        return torch.exp(-torch.abs(x - self.__media) / self.__beta) / (2 * self.__beta)

    def log(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        return -torch.log(2 * self.__beta) - torch.abs(x - self.__media) / self.__beta

    def grad_log(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        diff = x - self.__media
        return -torch.sign(diff) / self.__beta
    

class InverseGamma:
    def __init__(self, alpha, beta, device=None, dtype=torch.float64):
        # resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.__device = device
        self.__dtype = dtype

        # parâmetros como tensores
        self.__alpha = torch.as_tensor(alpha, device=self.__device, dtype=self.__dtype)
        self.__beta  = torch.as_tensor(beta,  device=self.__device, dtype=self.__dtype)

    def pdf(self, x):
        return torch.exp(self.log(x))

    def log(self, x):
        alpha = self.__alpha
        beta = self.__beta

        # evitar log(0) / divisão por zero
        eps = torch.finfo(x.dtype).eps
        x = torch.clamp(x, min=eps)

        return (
            alpha * torch.log(beta)
            - torch.lgamma(alpha)
            - (alpha + 1) * torch.log(x)
            - beta / x
        )

    def grad_log(self, x):
        alpha = self.__alpha
        beta = self.__beta

        eps = torch.finfo(x.dtype).eps
        x = torch.clamp(x, min=eps)

        return (beta - (alpha + 1) * x) / (x ** 2)
    
    def sample(self, size=1):
        # Gamma(alpha, scale = 1/beta)
        gamma_dist = torch.distributions.Gamma(self.__alpha, 1.0 / self.__beta)
        y = gamma_dist.sample((size,)).to(dtype=self.__dtype, device=self.__device)
        return 1.0 / y


class Normal:
    def __init__(self, media=0, sigma=1, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.__media = torch.tensor(media, dtype=self.__dtype, device=self.__device)
        self.__sigma = torch.tensor(sigma, dtype=self.__dtype, device=self.__device)
        self.__logZ = torch.log(2 * torch.pi * (self.__sigma ** 2))

    def pdf(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        return torch.exp(-(x - self.__media) ** 2 / (2 * self.__sigma ** 2)) / torch.sqrt(2 * torch.pi * (self.__sigma ** 2))

    def log(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        return -0.5 * (self.__logZ + (x - self.__media) ** 2 / (self.__sigma ** 2))

    def grad_log(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        return -(x - self.__media) / (self.__sigma ** 2)
    
    def sample(self, size=1):
        return torch.normal(self.__media, self.__sigma, size=(size,))


class MultivariateNormal:
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor | None = None, precision: torch.Tensor | None = None, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.mean = mean.to(device=self.__device, dtype=self.__dtype)

        if cov is None and precision is None:
            raise ValueError("Provide either cov or precision matrix.")

        if cov is not None:
            self.cov = cov.to(device=self.__device, dtype=self.__dtype)
            self.prec = torch.linalg.inv(self.cov)
        else:
            self.prec = precision.to(device=self.__device, dtype=self.__dtype)
            self.cov = torch.linalg.inv(self.prec)

        self.d = self.mean.shape[0]
        self.logdet_cov = torch.logdet(self.cov)
        self.logZ = -0.5 * (self.d * np.log(2 * np.pi) + self.logdet_cov)

    def log(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        diff = x - self.mean
        expo = -0.5 * torch.sum(diff * (self.prec @ diff.T).T, dim=-1)
        return self.logZ + expo

    def pdf(self, x: torch.Tensor):
        return torch.exp(self.log(x))

    def grad_log(self, x: torch.Tensor):
        x = x.to(device=self.__device, dtype=self.__dtype)
        diff = x - self.mean
        return -(self.prec @ diff.unsqueeze(-1)).squeeze(-1)


class PosterioriNormal:
    def __init__(self, X, y, priori, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.X = X.to(device=self.__device, dtype=self.__dtype)
        self.y = y.to(device=self.__device, dtype=self.__dtype)
        self.priori = priori

    def log_likelihood(self, theta, sigma=1.0):
        theta = theta.to(device=self.__device, dtype=self.__dtype)
        resid = self.y - self.X @ theta
        n = self.y.shape[0]
        return -0.5 * torch.sum(resid ** 2) / (sigma ** 2) - 0.5 * n * torch.log(2 * torch.pi * sigma ** 2)

    def log(self, theta):
        theta = theta.to(device=self.__device, dtype=self.__dtype)
        return self.log_likelihood(theta) + self.priori.log(theta)

    def grad_log_posteriori(self, theta, sigma=1.0):
        theta = theta.to(device=self.__device, dtype=self.__dtype)
        resid = self.y - self.X @ theta
        grad = (self.X.T @ resid) / (sigma ** 2)
        return grad + self.priori.grad_log(theta)

    def posteriori(self, theta):
        theta = theta.to(device=self.__device, dtype=self.__dtype)
        return torch.exp(self.log_likelihood(theta)) * self.priori.pdf(theta)


class PosterioriFromX:
    def __init__(self, X, priori, sigma_x=1.0, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.X = X.flatten().to(device=self.__device, dtype=self.__dtype)
        self.priori = priori
        self.sigma_x = torch.tensor(sigma_x, device=self.__device, dtype=self.__dtype)

    def log_likelihood(self, theta):
        theta = torch.as_tensor(theta, dtype=self.__dtype, device=self.__device)
        resid = self.X - theta
        return -(torch.sum(resid ** 2)) / 2

    def log(self, theta: torch.Tensor):
        theta = theta.to(device=self.__device, dtype=self.__dtype)
        return self.priori.log(theta) + self.log_likelihood(theta)

    def posteriori(self, theta):
        theta = theta.to(device=self.__device, dtype=self.__dtype)
        return torch.exp(self.log_likelihood(theta)) * self.priori.pdf(theta)