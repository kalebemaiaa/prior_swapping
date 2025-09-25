# Posteriors Inference Prior Swapping

Este projeto implementa métodos de amostragem bayesiana para explorar posterioris, com foco em **Metropolis-Hastings (MH)**, **Hamiltonian Monte Carlo (HMC)** e **post-inference prior swapping**.

## Usage 

Necessário ter git e python instalado. Clone o repositório e navegue até ele:

```bash
git clone https://github.com/kalebemaiaa/prior_swapping.git
cd prior_swapping
```

Opcionalmente cria uma venv e inicie ela:

```bash
python -m venv venv
.\venv\Scripts\activate
```

Instale as depêndencias

```bash
pip install -r requirements.txt
```

Agora basta usar os métodos implementados. Para amostragem de distribuições, a classe [Sampler](./auxiliar/SamplerMCMC.py) é útil siga os exemplos de uso:

```python
def likelihood_normal(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x**2 / 2) / torch.sqrt(torch.tensor(2 * torch.pi))

# stepsize bem grande!!
def proposal_distribution(x: torch.Tensor, stepsize: float = 50) -> torch.Tensor:
    return torch.normal(mean=x, std=stepsize)

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

def log_prob_laplace(x):
    b = 1.0
    return -np.log(2 * b) - torch.abs(x) / b

def grad_log_likelihood_laplace(x: torch.Tensor) -> torch.Tensor:
    b = 1.0
    return -torch.sign(x) / b

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
```

Ou amostrar e usar prior swapping, como [feito em main](./main.py) através de $\pi_f(\theta)\sim\mathcal{N}(0,1)$, $\pi(\theta)\sim Laplace(0, 0.01)$ e $P(Y|\theta, 1)\sim\mathcal{N}(\theta, 1)$

## Desenvolvimento

- ✅[Amostragem via MH e HMC (target & false posterior inference)](./auxiliar/SamplerMCMC.py#L24) 
- ✅[False posterior IS & Prior swap exact](./main.py#L119)
- [Otimização de alpha](./auxiliar/otimizador_alpha.py) para o desenvolvimento de [Prior swap parametric & Prior swap IS]()
- [Prior swap semiparametric]()

## Referências

- [Post-Inference Prior Swapping](https://arxiv.org/pdf/1606.00787)
- [Provable benefits of score matching](https://arxiv.org/pdf/2306.01993)
- [HMC implementing MCMC](https://www.tcbegley.com/blog/posts/mcmc-part-2)
- [Score-models implementations](https://github.com/cheind/score-matching)
- [Score matching explicação](https://jaketae.github.io/study/sliced-score-matching/)

