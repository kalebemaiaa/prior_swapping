from abc import ABC, abstractmethod
import numpy as np
import torch

class Distribution(ABC):
    @abstractmethod
    def pdf(self, x: torch.Tensor) -> torch.Tensor: ...
    
    @abstractmethod
    def log(self, x: torch.Tensor) -> torch.Tensor: ...
    
class InverseGamma(Distribution):
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


class Normal(Distribution):
    def __init__(self, media=0, sigma:float|int=1, device = None, dtype = torch.float64):
        self.__device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.__dtype = dtype
        self.__media = torch.tensor(media, dtype=self.__dtype, device=self.__device)
        self.__sigma = torch.tensor(sigma, dtype=self.__dtype, device=self.__device)
        self.__logZ = torch.log(2 * torch.pi * (self.__sigma ** 2))

    def pdf(self, x: torch.Tensor):
        return torch.exp(-(x - self.__media) ** 2 / (2 * self.__sigma ** 2)) / torch.sqrt(2 * torch.pi * (self.__sigma ** 2))

    def log(self, x: torch.Tensor):
        return -0.5 * (self.__logZ + (x - self.__media) ** 2 / (self.__sigma ** 2))

    def grad_log(self, x: torch.Tensor):
        return -(x - self.__media) / (self.__sigma ** 2)
    
    def sample(self, size=1):
        return torch.normal(self.__media.item(), self.__sigma.item(), size=(size,))


class NormalVarianciaDesconhecida:
    def __init__(self, mean: float|int) -> None:
        self.__mean = mean
        
    def pdf(self, x, variancia):
        pass
        
    def log(self, x, variancia):
        pass
    
    def grad_log(self, x, variancia):
        pass


### Classes para usar em todos métodos

class ParametricFakeDensity:
    def __init__(self, fake_priori: Distribution, data_density, alpha_star: torch.Tensor, n_dataset: int) -> None:
        self.__fake_priori = fake_priori
        self.__data_density = data_density
        self.__alpha_star = alpha_star
        self.__ratio = n_dataset / alpha_star.shape[0]
        
    
    def pdf(self, x):
        prod = 1
        for alpha_i in self.__alpha_star:
            prod *= self.__data_density.pdf(alpha_i, x) ** self.__ratio
        return self.__fake_priori.pdf(x) * prod
    
    def log(self, x):
        soma = 0
        for alpha_i in self.__alpha_star:
            soma += self.__data_density.log(alpha_i, x)
        return self.__fake_priori.log(x) + self.__ratio * soma
    
    
class SemiParametricFakeDensity:
    def __init__(self, samples_theta, kernel_function: Callable) -> None:
        self.__samples_theta = samples_theta
        self.__num_sampels = samples_theta.shape[0]
    
    def pdf(self):
        somatorio = 0
        for theta_i in self.__samples_theta:
            kernel_function(\|samples_theta - theta\| / self.__bandwidth) * fake_parametric(\theta) / fake_parametric(theta_i) * (1 / (bandwidth ** dimensao de theta))
        return 1/self.__num_sampels * 
    
    def log(self, )

    
### DISTRIBUIÇÕES CASO REGRESSÃO
class Laplace(Distribution):
    def __init__(self, media: float = 10, beta: float = 1, device = None, dtype = torch.float64):
        self.__device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.__dtype = dtype
        self.__media = torch.tensor(media, dtype=self.__dtype, device=self.__device)
        self.__beta = torch.tensor(beta, dtype=self.__dtype, device=self.__device)

    def pdf(self, x: torch.Tensor):
        return torch.exp(-torch.abs(x - self.__media) / self.__beta) / (2 * self.__beta)

    def log(self, x: torch.Tensor):
        return -torch.log(2 * self.__beta) - torch.abs(x - self.__media) / self.__beta

    def grad_log(self, x: torch.Tensor):
        diff = x - self.__media
        return -torch.sign(diff) / self.__beta


class MultivariateNormal(Distribution):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor | None = None, precision: torch.Tensor | None = None, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.mean = mean.to(device=self.__device, dtype=self.__dtype)

        if cov is None and precision is None:
            raise ValueError("Provide either cov or precision matrix.")

        if cov is not None:
            self.cov = cov.to(device=self.__device, dtype=self.__dtype)
            self.prec = torch.linalg.inv(self.cov)
        elif precision is not None:
            self.prec = precision.to(device=self.__device, dtype=self.__dtype)
            self.cov = torch.linalg.inv(self.prec)
        else:
            raise ValueError("O argumento `cov` ou o argumento `precision` precisam ser passados")

        self.d = self.mean.shape[0]
        self.logdet_cov = torch.logdet(self.cov)
        self.logZ = -0.5 * (self.d * np.log(2 * np.pi) + self.logdet_cov)

    def log(self, x: torch.Tensor):
        diff = x - self.mean
        expo = -0.5 * torch.sum(diff * (self.prec @ diff.T).T, dim=-1)
        return self.logZ + expo

    def pdf(self, x: torch.Tensor):
        return torch.exp(self.log(x))

    def grad_log(self, x: torch.Tensor):
        diff = x - self.mean
        return -(self.prec @ diff.unsqueeze(-1)).squeeze(-1)
    
class PosterioriNormal(Distribution):
    def __init__(self, X, y, priori, device = None, dtype = torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.X = X
        self.y = y
        self.priori = priori

    def log_likelihood(self, theta, sigma=1.0):
        resid = self.y - self.X @ theta
        n = self.y.shape[0]
        return -0.5 * torch.sum(resid ** 2) / (sigma ** 2) - 0.5 * n * torch.log(2 * torch.pi * sigma ** 2) # type: ignore

    def log(self, x):
        return self.log_likelihood(x) + self.priori.log(x)

    def grad_log_posteriori(self, theta, sigma=1.0):
        resid = self.y - self.X @ theta
        grad = (self.X.T @ resid) / (sigma ** 2)
        return grad + self.priori.grad_log(theta)

    def posteriori(self, theta):
        return torch.exp(self.log_likelihood(theta)) * self.priori.pdf(theta)