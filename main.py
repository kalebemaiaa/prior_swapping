from auxiliar.distribuicoes import Laplace, Normal, PosterioriNormal, PosterioriFromX
from auxiliar.SamplerMCMC import Sampler
from auxiliar.utils import f_next_state, sample_false_posterior_linear_regression, get_weights_IS, make_f_verossimilhanca

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import kagglehub
import torch
import os

sns.set_theme()

#@title Download dataset
# Download latest version
path = kagglehub.dataset_download("zakariaeyoussefi/song-year-prediction-msd")

complete_path = os.path.join(path, "YearPredictionMSD.csv")
df_yearMSD = pd.read_csv(complete_path)

# Convertendo dados para tensores
columns_features = [c for c in df_yearMSD.columns if c != "Year"]

x_data = df_yearMSD[columns_features]
y_data = df_yearMSD["Year"]

# converte para tensor
X = torch.tensor(x_data.values, dtype=torch.float64)
y = torch.tensor(y_data.values, dtype=torch.float64)

# adicionando intercepto
ones = torch.ones((X.shape[0], 1), dtype=torch.float64)
X = torch.cat([ones, X], dim=1) 

N, D = X.shape

# use_log_pdf, f_next_state := argumentos mh function
# step_size, n_steps, f_grad_log_prob := argumentos hmc function
main_sampler = Sampler()

# TODO: dado um conjunto de amostras, reamostrar via importance sampling:
# criar conjunto de amostras da normal: priori conjugada
# normal_padrao = Normal()
# posteriori_normal = PosterioriNormal(X, y, priori=normal_padrao)

# initial_state = torch.zeros(D, dtype=torch.float64)
# main_sampler.f_log_prob = posteriori_normal.log_posteriori
# false_posteriori_samples = main_sampler.get_samples(
#     method="hmc",
#     estado_inicial=initial_state,
#     n_amostras=10,
#     burnin=0,
#     step_size=0.01,
#     n_steps=100,
#     f_grad_log_prob=posteriori_normal.grad_log_posteriori
# )

# print(false_posteriori_samples)

# main_sampler.f_log_prob = posteriori_normal.log_posteriori
# false_posteriori_samples = main_sampler.get_samples(
#     method="mh",
#     estado_inicial=initial_state,
#     n_amostras=1000,
#     burnin=0,
#     use_log_pdf = True,
#     f_next_state = f_next_state
# )

# print(false_posteriori_samples)

# Teste figura 1/2

def proposal_distribution(theta_curr, proposal_std=10, **kwargs):
    noise = torch.normal(mean=torch.zeros_like(theta_curr), std=proposal_std)
    return theta_curr + noise

theta_true = 8
n = 5
X_teste = torch.normal(mean=theta_true, std=1.0, size=(n,))
initial_state_teste = torch.as_tensor(0.0, dtype=torch.float64)

# plot posteriori normal
normal_padrao = Normal()
posteriori_x = PosterioriFromX(X_teste, priori=normal_padrao)
main_sampler.f_log_prob = posteriori_x.log_posteriori
amostras_teste_normal = main_sampler.get_samples(
    method="mh",
    estado_inicial=initial_state_teste,
    n_amostras=1000,
    burnin=0,
    use_log_pdf=True,  # importante usar log para estabilidade
    f_next_state=proposal_distribution
)
sns.kdeplot(amostras_teste_normal, label = "False posterior samples", color="red")

# plot_laplace posteriori amostras
laplace_f1 = Laplace(media=10, beta=1/4)
posteriori_x = PosterioriFromX(X_teste, priori=laplace_f1)
main_sampler.f_log_prob = posteriori_x.log_posteriori
amostras_teste_laplace = main_sampler.get_samples(
    method="mh",
    estado_inicial=initial_state_teste,
    n_amostras=1000,
    burnin=0,
    use_log_pdf=True,  # importante usar log para estabilidade
    f_next_state=proposal_distribution
)
sns.kdeplot(amostras_teste_laplace, label="Posterior samples", color="blue")

# plota distribuicoes
normal_grid = torch.linspace(-4, 4, 1000)
normal_pdf_values = normal_padrao.pdf(normal_grid)
laplace_grid = torch.linspace(5, 15, 1000)

laplace_pdf_values = laplace_f1.pdf(laplace_grid)

pesos = get_weights_IS(amostras_teste_normal, normal_padrao.log, laplace_f1.log)
print(pesos.sum())
mean_hat_IS = pesos.dot(amostras_teste_normal).mean()

mean_real = amostras_teste_laplace.mean()

sns.lineplot(x=laplace_grid, y=laplace_pdf_values, color="#E038D5", linestyle="--", label="PI(theta)")
sns.lineplot(x=normal_grid, y=normal_pdf_values, color="green", linestyle="--", label="PI_f(theta)")
plt.axhline(y=0)
plt.xlim([-3, 13])
plt.ylim([-0.5, 2])
plt.ylabel("Densidade")
plt.legend()
plt.savefig("./img/figure_2.png")
plt.title(f"mean_hat = {mean_hat_IS} vs mean_real = {mean_real}")
plt.show()

