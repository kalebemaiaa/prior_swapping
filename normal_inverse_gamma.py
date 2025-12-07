from auxiliar.distribuicoes import Normal, InverseGamma
from auxiliar.otimizador_alpha import Otimizador
from auxiliar.utils import make_log_density_SM

import matplotlib.pyplot as plt
import seaborn as sns

import torch

sns.set_theme()
sns.set_style()

# X1,...,XN normal (0, sigma^2), variância desconhecida
# amostrar de normal(0, epsilon) com epsilon pequeno
# no caso, sigma = \sqrt{\theta}
# poderiamos modelar a priori target como normal 0, 1
# mas vamos usar \theta como InverseGamma(alpha, beta)

# Gerando amostras x_i, com normal(0, 0.37)
real_distribuicao = Normal(0, 0.37)
data = real_distribuicao.sample(500000)

# print(data)
# sns.kdeplot(data)
# plt.show()

# PRIORIS
target_priori = Normal(media=0, sigma=1) # temos uma normal(0,1)
alpha_0 = 2
beta_0 = 1
fake_priori = InverseGamma(alpha=alpha_0, beta=beta_0) # Temos uma Inverse-Gamma(alpha, beta)

# POSTERIORIS;
# amostar da fake conjugada Normal-Inverse-Gamma
# se temos n datapoins x_i, então a posteriori é da forma:
#  IG(a_0 + n/2, b_0+\sum (x_i)^2/2)
n_samples = data.shape[0]
alpha_post = alpha_0 + (n_samples / 2)
beta_post = beta_0 + data.pow(2).sum()/2
fake_posteriori = InverseGamma(alpha=alpha_post, beta=beta_post)
fake_samples = fake_posteriori.sample(1000)

# # Vai receber um alpha, theta, n, k e retornar log para ser usado no score matching
def log_likelihood_N(alpha, theta):
    return -0.5 * (torch.log(2 * torch.pi * (theta ** 2)) - alpha**2/(theta**2))
    
log_density_SM = make_log_density_SM(log_fake_prior=fake_priori.log, log_likelihood=log_likelihood_N) 

# Aplicando otimizador para encontrar alpha optimal
fake_samples = fake_samples.unsqueeze(1)
otimizador_IG = Otimizador(use_hutchinson=True)
alpha_star = otimizador_IG.find_alpha_star(
    log_density_SM=log_density_SM, 
    samples=fake_samples,
    N_dataset=len(data),
    n_steps=1000,
    K_alpha_samples=1000,
    alpha_dim=1
)

# # Uma vez que temos alpha_star, podemos calcular p(theta)
# def wrapper_p_theta()
# def p_theta(theta):
#     return 