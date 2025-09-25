import torch

class Laplace:
    def __init__(self, media: float = 10, beta: float = 1):
        self.__media = torch.as_tensor(media)
        self.__beta = torch.as_tensor(beta)

    def pdf(self, x: torch.Tensor):
        return torch.exp(-torch.abs(x - self.__media) / self.__beta) / (2 * self.__beta)

    def log(self, x: torch.Tensor):
        return -torch.log(2 * self.__beta) - torch.abs(x - self.__media) / self.__beta
    
    def grad_log(self, x: torch.Tensor) -> float:
        # diff / (b * abs(diff))
        diff = (x - self.__media)
        return - torch.sign(diff) / self.__beta

class Normal:
    def __init__(self, media = 0, sigma = 1):
        self.__media = torch.as_tensor(media)
        self.__sigma = torch.as_tensor(sigma)

    def pdf(self, x):
        return torch.exp(-(x - self.__media) ** 2 / (2 * self.__sigma ** 2)) / torch.sqrt(2 * torch.pi * (self.__sigma ** 2))

    def log(self, x):
        return -0.5 * torch.sum(torch.log(2 * torch.pi * (self.__sigma ** 2)) + (x-self.__media) ** 2 / (self.__sigma ** 2))
    
    def grad_log(self, x):
        return -(x - self.__media)/(self.__sigma**2)

class PosterioriNormal:
    def __init__(self, X, y, priori, sigma=None):
        self.X = torch.as_tensor(X, dtype=torch.float64)
        self.y = torch.as_tensor(y, dtype=torch.float64)
        self.priori = priori

        if sigma:
            self.__sigma = torch.as_tensor(sigma, dtype=torch.float64)
        else:
            self.__sigma = sigma
        

    def log_likelihood(self, theta):
        if self.__sigma is None:
            residuo = self.y - self.X @ theta
            len_amostras = self.y.shape[0]
            sigma = torch.sqrt(torch.sum(residuo**2) / (len_amostras - self.X.shape[1]))
        else:
            sigma = self.__sigma

        resid = self.y - self.X @ theta
        n = self.y.shape[0]
        return -0.5 * torch.sum(resid**2) / (sigma**2) - 0.5 * n * torch.log(2*torch.pi*sigma**2)

    def log_posteriori(self, theta):
        return self.log_likelihood(theta) + self.priori.log(theta)

    def grad_log_posteriori(self, theta):
        if self.__sigma is None:
            residuo = self.y - self.X @ theta
            len_amostras = self.y.shape[0]
            sigma = torch.sqrt(torch.sum(residuo**2) / (len_amostras - self.X.shape[1]))
        else:
            sigma = self.__sigma

        resid = self.y - self.X @ theta
        grad = (self.X.T @ resid) / (sigma**2)

        return grad + self.priori.grad_log(theta)

    def posteriori(self, theta):
        return torch.exp(self.log_likelihood(theta)) * self.priori.pdf(theta)
    

class PosterioriFromX:
    def __init__(self, X, priori,sigma_x=1.0):
        self.X = X.flatten()
        self.n = self.X.shape[0]
        self.priori = priori
        self.sigma_x = torch.as_tensor(sigma_x)

    def log_likelihood(self, theta):
        # garante que theta tem dtype/shape compatível
        theta = torch.as_tensor(theta, dtype=torch.float64)
        resid = self.X - theta
        return -(self.X.shape[0]* torch.log(2 * torch.pi * (self.sigma_x**2)) + torch.sum(resid**2)) / 2

    def log_posteriori(self, theta: torch.Tensor):
        return self.priori.log(theta) + self.log_likelihood(theta)

    def posteriori(self, theta):
        return torch.exp(self.log_likelihood(theta)) * self.priori.pdf(theta)