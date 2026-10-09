# Projeto 2 — Etapa 1: Explorando os atrasos de entrega
# Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
# Arquivos esperados (na mesma pasta deste arquivo):
#   - olist_orders_dataset.csv
#   - olist_customers_dataset.csv
#
# OBJETIVO: entender quantos pedidos atrasam, quando e para onde,
# antes de pensar em qualquer modelo.

# %% [markdown]
# # Projeto 2 — Previsão de atraso de entrega
# **Autor:** Rafael
# **Pergunta:** este pedido vai atrasar em relação à data prometida ao cliente?
# **Etapa 1:** conhecer os dados de atraso.

# %% Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (11, 5)

# %% Carregar os dados
pedidos = pd.read_csv("olist_orders_dataset.csv")
clientes = pd.read_csv("olist_customers_dataset.csv")

print("Pedidos:", pedidos.shape)
print("Clientes:", clientes.shape)

# %% Converter as colunas de data
colunas_data = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for coluna in colunas_data:
    pedidos[coluna] = pd.to_datetime(pedidos[coluna])

# %% [markdown]
# ## Ficar só com os pedidos entregues
# Só dá pra saber se atrasou quando o pedido foi entregue.
# Pedidos cancelados ou ainda em trânsito não entram nesta análise.

# %% Filtrar pedidos entregues com as duas datas preenchidas
entregues = pedidos[
    (pedidos["order_status"] == "delivered")
    & pedidos["order_delivered_customer_date"].notna()
    & pedidos["order_estimated_delivery_date"].notna()
].copy()

print("Pedidos entregues com as duas datas:", len(entregues))

# %% [markdown]
# ## Criar a resposta: atrasou ou não
# Comparamos só o DIA (sem a hora). Se a entrega real foi depois do dia
# prometido, o pedido atrasou (1). Senão, chegou no prazo (0).

# %% Criar a coluna "atrasou"
entregue_dia = entregues["order_delivered_customer_date"].dt.normalize()
prometido_dia = entregues["order_estimated_delivery_date"].dt.normalize()

entregues["atrasou"] = (entregue_dia > prometido_dia).astype(int)
entregues["dias_de_atraso"] = (entregue_dia - prometido_dia).dt.days

# %% Quantos pedidos atrasam?
total = len(entregues)
atrasados = entregues["atrasou"].sum()
taxa = atrasados / total * 100

print(f"Pedidos atrasados: {atrasados} de {total}")
print(f"Taxa de atraso: {taxa:.1f}%")

entregues["atrasou"].value_counts().rename({0: "No prazo", 1: "Atrasou"}).plot(
    kind="bar", title="Pedidos no prazo x atrasados"
)
plt.ylabel("Nº de pedidos")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Atraso ao longo do tempo
# A taxa de atraso muda conforme o mês? Datas de pico (como a Black Friday)
# costumam apertar a operação.

# %% Taxa de atraso por mês
entregues["mes_compra"] = entregues["order_purchase_timestamp"].dt.to_period("M")
taxa_mes = entregues.groupby("mes_compra")["atrasou"].mean() * 100

taxa_mes.plot(kind="bar", title="Taxa de atraso por mês da compra (%)")
plt.ylabel("% de pedidos atrasados")
plt.xlabel("Mês da compra")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Atraso por estado do cliente
# Para saber o estado, juntamos a tabela de pedidos com a de clientes
# pela coluna customer_id (é como um PROCV do Excel).

# %% Juntar com clientes e calcular a taxa por estado
entregues = entregues.merge(
    clientes[["customer_id", "customer_state"]], on="customer_id", how="left"
)

taxa_estado = (
    entregues.groupby("customer_state")["atrasou"].mean() * 100
).sort_values(ascending=False)

taxa_estado.plot(kind="bar", title="Taxa de atraso por estado do cliente (%)")
plt.ylabel("% de pedidos atrasados")
plt.xlabel("Estado")
plt.tight_layout()
plt.show()

print("Estados com MAIS atraso:")
print(taxa_estado.head(5).round(1))
print("\nEstados com MENOS atraso:")
print(taxa_estado.tail(5).round(1))

# %% [markdown]
# ## Quando atrasa, atrasa quanto?

# %% Dias de atraso entre os pedidos atrasados
so_atrasados = entregues[entregues["atrasou"] == 1]
print("Dias de atraso (pedidos atrasados):")
print(so_atrasados["dias_de_atraso"].describe().round(1))

# %% [markdown]
# ## Minhas observações da Etapa 1
#
# 1. A taxa de atraso é de 6,8% (6.534 de 96.470 pedidos entregues). É baixa
#    no geral, mas isso deixa os dados desbalanceados: um modelo que sempre
#    chuta "não atrasa" acerta 93,2% sem aprender nada. Por isso, além da
#    acurácia, vamos usar precisão e recall.
#
# 2. A taxa de atraso teve picos em novembro de 2017 (cerca de 12%, mês da
#    Black Friday), fevereiro de 2018 (cerca de 14%) e março de 2018 (cerca
#    de 19%, o maior do período). Depois de abril de 2018, voltou ao normal.
#    A barra de setembro de 2016 (100%) foi ignorada: é o início da base,
#    com pouquíssimos pedidos.
#
# 3. Os estados que mais atrasam são todos do Nordeste: Alagoas (21,4%),
#    Maranhão, Sergipe, Piauí e Ceará, que são os maiores ofensores. O motivo
#    provável é a distância: a maioria dos vendedores fica no Sudeste, e quanto
#    mais longe, mais etapas de transporte e mais chance de atraso.
