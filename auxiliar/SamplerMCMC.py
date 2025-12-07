from typing import Callable, Literal, overload
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# @title Metropolis Hastings implementação
class Sampler:
    def __init__(self, f_log_prob: Callable |None = None, device = None, dtype=torch.float64):
        self.__device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.__dtype = dtype
        self.__f_log_prob = f_log_prob
        self.amostras = None

    @property
    def f_log_prob(self):
        return self.__f_log_prob

    @f_log_prob.setter
    def f_log_prob(self, f_log_prob):
        self.__f_log_prob = f_log_prob

    def __mh_updater(self, estado_atual: torch.Tensor, logp_atual: float, use_log_pdf: bool, f_next_state: Callable|None = None):
        was_scalar = False
        if estado_atual.ndim == 0:
            estado_atual = estado_atual.unsqueeze(-1)
            was_scalar = True
        if not f_next_state:
            dim = estado_atual.shape[-1]
            cov = torch.eye(dim, device=estado_atual.device, dtype=estado_atual.dtype)
            f_next_state = lambda x: torch.distributions.MultivariateNormal(x, covariance_matrix=cov).sample()
        estado_proposto = f_next_state(estado_atual)

        # verossimilhanca
        logp_proposto = self.__f_log_prob(estado_proposto)
        # threshold atual
        threshold = torch.rand((), device=self.__device, dtype=self.__dtype)
        if use_log_pdf:
            threshold = threshold.log()

            # # --- > if (log(verossimilhanca_propost / verossimilhanca_atual)) > log(threshcold)
            # # --- > if (log(verossimilhanca_propost) - log(verossimilhanca_atual)) > log(threshcold
            # print("proposto: ", logp_proposto)
            # print("atual: ", logp_atual)
            # print(f"threshold {threshold} < {logp_proposto - logp_atual} proposto - atual ==>  ", threshold < logp_proposto - logp_atual)
            # print(threshold)
            # print("-"*25)
            if threshold < logp_proposto - logp_atual:
                return estado_proposto, logp_proposto

            return estado_atual, logp_atual

        aceitacao_raio = logp_proposto / logp_atual
        if threshold < min(aceitacao_raio, 1):
            return estado_proposto, logp_proposto

        return estado_atual, logp_atual

    def __hmc_updater(self, estado_atual: torch.Tensor, logp_atual: float, step_size, n_steps: int, f_grad_log_prob: Callable):
        r0 = torch.randn_like(estado_atual, device= self.__device, dtype=self.__dtype)

        valor_inicial = logp_atual - 0.5 * torch.sum(r0**2)

        #leapfrog_integration
        # https://www.tcbegley.com/blog/posts/mcmc-part-2
        theta = estado_atual.clone()
        r = r0.clone()

        r = r + 0.5 * step_size * f_grad_log_prob(theta)
        for _ in range(n_steps - 1):
            theta = theta + step_size * r

            grad = f_grad_log_prob(theta)
            r = r + step_size * grad

        r = r - 0.5 * step_size * grad

        logp_proposto = self.__f_log_prob(theta)

        valor_final = logp_proposto - 0.5 * torch.sum(r ** 2)

        log_accept_ratio = valor_final - valor_inicial
        threshold = torch.rand((), device=self.__device, dtype=self.__dtype).log()
        if threshold < log_accept_ratio:
            return theta, logp_proposto

        return estado_atual, logp_atual

    @overload
    def get_samples(self, method: Literal["mh"], estado_inicial: torch.Tensor, use_log_pdf: bool, n_amostras: int = 1000, burnin: float = 0.2, f_next_state: Callable|None = None) -> torch.Tensor: ...

    @overload
    def get_samples(self, method: Literal["hmc"], estado_inicial: torch.Tensor, step_size, n_steps: int, f_grad_log_prob: Callable, n_amostras: int = 1000, burnin: float = 0.2) -> torch.Tensor: ...

    def get_samples(self, method: str, estado_inicial: torch.Tensor, n_amostras: int=1000, burnin: float=0.2, **kwargs):
        if method.lower() == "mh":
            updater = self.__mh_updater
        elif method.lower() == "hmc":
            updater = self.__hmc_updater
        else:
            raise ValueError("method deve ser 'hmc' ou 'mh'")

        if self.__f_log_prob is None:
            raise ValueError("f_log_prob é um argumento necessário para a execução de qualquer método")

        estado_atual = torch.as_tensor(estado_inicial, device=self.__device, dtype=self.__dtype)
        logp_atual = self.__f_log_prob(estado_atual)

        burnin_amostras = int(burnin * n_amostras)
        amostras = []

        for _ in tqdm(range(n_amostras), desc="Gerando amostras"):
            estado_atual, logp_atual = updater(estado_atual, logp_atual, **kwargs)

            if _ > burnin_amostras:
                amostras.append(estado_atual.clone())

        self.amostras = torch.stack(amostras)
        return self.amostras


    def plot_cadeia(self, amostras=None, burnin: float|None=None, save_path: str|None=None, show_img: bool = False):
        if amostras is None:
            if self.amostras is None:
                raise ValueError("O valor de amostras deve ser passado ou gerado através de um método de amostragem")

            amostras = self.amostras

        amostras = torch.as_tensor(amostras).flatten()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        if isinstance(burnin, float):
            n_burnin = int(burnin * len(amostras))
            amostras_burnin = amostras[:n_burnin]
            sns.lineplot(ax=axes[0], x=[i + 1 for i in range(len(amostras_burnin))], y=amostras_burnin, label="Samples burnin")

            amostras_real = amostras[n_burnin:]
            x_real = [len(amostras_burnin) + i + 1 for i in range(len(amostras_real))]
        else:
            amostras_real = amostras
            x_real = [i + 1 for i in range(len(amostras_real))]

        sns.lineplot(ax=axes[0], x=x_real, y=amostras_real, label="Samples aceitas")
        axes[0].set_title('Lineplot')

        sns.histplot(amostras_real, ax=axes[1], kde=True) #-> aqui poderiamos plotar só o KDE
        axes[1].set_title('Histplot')

        plt.tight_layout()
        if isinstance(save_path, str):
            try:
                plt.savefig(save_path)
            except Exception as e:
                print("Erro ao tentar salvar imagem!")
                print(e)

        if show_img:
            plt.show()
        

if __name__ == "__main__":
    # NOTE: EXEMPLO DE AMOSTRAGEM DE UMA NORMAL PADRAO
    def likelihood_normal(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x**2 / 2) / torch.sqrt(torch.tensor(2 * torch.pi))

    # aqui coloquei stepsize grande para andar muit -> antes estava tendo muito amostra aceita
    def proposal_distribution(x: torch.Tensor, stepsize: float = 50) -> torch.Tensor:
        return torch.normal(mean=x, std=stepsize)
    
    def likelihood_laplace(x: torch.Tensor) -> torch.Tensor:
        b = 1.0
        return 1/(2*b) * torch.exp(-torch.abs(x)/b)

    # como escolher essa função ainda é um enigma :p
    def proposal_laplace(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        u = torch.rand_like(x) - 0.5
        return x - scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
    
    def log_prob_normal(x):
        return -0.5 * x**2 - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    
    def log_prob_laplace(x):
        b = 1.0
        return -np.log(2 * b) - torch.abs(x) / b
    
    def grad_log_likelihood_normal(x: torch.Tensor):
        return - x
    
    def grad_log_likelihood_laplace(x: torch.Tensor) -> torch.Tensor:
        b = 1.0
        return -torch.sign(x) / b


    initial_state = torch.as_tensor(0, dtype=torch.float64)
    num_samples = int(1e4)
    burnin=0
    
    sampler_teste = Sampler()

    # plot normal mh metodo
    sampler_teste.f_log_prob = likelihood_normal
    amostras_teste = sampler_teste.get_samples(
        method="mh", 
        estado_inicial=initial_state, 
        n_amostras=num_samples, 
        burnin=burnin, 
        use_log_pdf=False, 
        f_next_state=proposal_distribution
    )
    sampler_teste.plot_cadeia(amostras_teste, burnin=0.2, save_path= "./img/mh_sample_normal.png")

    # plot laplace mh metodo
    sampler_teste.f_log_prob = likelihood_laplace
    amostras_teste = sampler_teste.get_samples(
        method="mh", 
        estado_inicial=initial_state, 
        n_amostras=num_samples, 
        burnin=burnin, 
        use_log_pdf=False, 
        f_next_state=proposal_laplace
    )
    sampler_teste.plot_cadeia(amostras_teste, burnin=0.2, save_path= "./img/mh_sample_laplace.png")

    # plot normal hmc method
    sampler_teste.f_log_prob = log_prob_normal
    amostras_teste = sampler_teste.get_samples(
        method="hmc", 
        estado_inicial=initial_state, 
        n_amostras=num_samples, 
        burnin=burnin, 
        step_size=0.01, 
        n_steps=100,
        f_grad_log_prob=grad_log_likelihood_normal
    )
    sampler_teste.plot_cadeia(amostras_teste, burnin=0.2, save_path= "./img/hmc_sample_normal.png")

    # plot laplace hmc method
    sampler_teste.f_log_prob = log_prob_laplace
    amostras_teste = sampler_teste.get_samples(
        method="hmc", 
        estado_inicial=initial_state, 
        n_amostras=num_samples, 
        burnin=burnin, 
        step_size=0.01, 
        n_steps=100,
        f_grad_log_prob=grad_log_likelihood_laplace
    )
    sampler_teste.plot_cadeia(amostras_teste, burnin=0.2, save_path= "./img/hmc_sample_laplace.png")