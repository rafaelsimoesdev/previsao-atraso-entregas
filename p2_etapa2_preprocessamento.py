# Projeto 2 — Etapa 2: Pré-processamento
# Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
# Arquivos esperados (na mesma pasta deste arquivo):
#   - olist_orders_dataset.csv
#   - olist_customers_dataset.csv
#   - olist_order_items_dataset.csv
#   - olist_products_dataset.csv
#
# OBJETIVO: juntar as tabelas, criar as pistas que existem NA HORA DA COMPRA,
# tratar nulos, transformar texto em número e salvar uma base limpa
# (base_modelo.csv) para treinar o modelo na Etapa 3.

# %% [markdown]
# # Projeto 2 — Etapa 2: Preparando os dados
# **Regra de ouro:** só usamos informações que existem no momento da compra.
# Usar dados do futuro (como a data em que o pedido saiu para a transportadora)
# seria "colar na prova": isso se chama vazamento de dados (data leakage).

# %% Importar bibliotecas
import pandas as pd

# %% Carregar as 4 tabelas
pedidos = pd.read_csv("olist_orders_dataset.csv")
clientes = pd.read_csv("olist_customers_dataset.csv")
itens = pd.read_csv("olist_order_items_dataset.csv")
produtos = pd.read_csv("olist_products_dataset.csv")

for nome, tabela in [("Pedidos", pedidos), ("Clientes", clientes),
                     ("Itens", itens), ("Produtos", produtos)]:
    print(f"{nome}: {tabela.shape[0]} linhas, {tabela.shape[1]} colunas")

# %% Converter datas e ficar só com pedidos entregues (igual à Etapa 1)
for coluna in ["order_purchase_timestamp",
               "order_delivered_customer_date",
               "order_estimated_delivery_date"]:
    pedidos[coluna] = pd.to_datetime(pedidos[coluna])

base = pedidos[
    (pedidos["order_status"] == "delivered")
    & pedidos["order_delivered_customer_date"].notna()
    & pedidos["order_estimated_delivery_date"].notna()
].copy()

# %% Criar a resposta (o que o modelo vai prever)
entregue_dia = base["order_delivered_customer_date"].dt.normalize()
prometido_dia = base["order_estimated_delivery_date"].dt.normalize()
base["atrasou"] = (entregue_dia > prometido_dia).astype(int)

# %% [markdown]
# ## Pistas de data e prazo
# O prazo prometido (dias entre a compra e a data prometida) é conhecido
# na hora da compra, então pode ser usado.

# %% Criar pistas de data
compra = base["order_purchase_timestamp"]
base["mes_compra"] = compra.dt.month
base["dia_semana_compra"] = compra.dt.dayofweek
base["prazo_prometido_dias"] = (prometido_dia - compra.dt.normalize()).dt.days

# %% [markdown]
# ## Juntar o estado do cliente (PROCV com a tabela de clientes)

# %% Estado do cliente
base = base.merge(clientes[["customer_id", "customer_state"]],
                  on="customer_id", how="left")

# %% [markdown]
# ## Pistas dos itens e produtos
# Um pedido pode ter vários itens. Primeiro juntamos cada item com o seu
# produto (para pegar peso e tamanho), depois somamos tudo por pedido.

# %% Peso e volume de cada produto
produtos["volume_cm3"] = (produtos["product_length_cm"]
                          * produtos["product_height_cm"]
                          * produtos["product_width_cm"])

itens = itens.merge(
    produtos[["product_id", "product_weight_g", "volume_cm3"]],
    on="product_id", how="left"
)

# %% Somar por pedido
por_pedido = itens.groupby("order_id").agg(
    qtd_itens=("order_item_id", "count"),
    valor_produtos=("price", "sum"),
    valor_frete=("freight_value", "sum"),
    peso_total_g=("product_weight_g", "sum"),
    volume_total_cm3=("volume_cm3", "sum"),
).reset_index()

base = base.merge(por_pedido, on="order_id", how="left")

# %% [markdown]
# ## Tratar dados faltando (nulos)
# Primeiro olhamos quantos nulos existem. Depois preenchemos os números
# com a MEDIANA (o valor do meio), que não é puxada por valores extremos.

# %% Ver e tratar os nulos
pistas_numericas = ["mes_compra", "dia_semana_compra", "prazo_prometido_dias",
                    "qtd_itens", "valor_produtos", "valor_frete",
                    "peso_total_g", "volume_total_cm3"]

# Peso ou volume igual a zero é dado faltando (ao somar, o pandas transforma
# "sem peso" em 0). Marcamos como nulo antes de contar.
for coluna in ["peso_total_g", "volume_total_cm3"]:
    base.loc[base[coluna] == 0, coluna] = None

print("Nulos ANTES do tratamento:")
print(base[pistas_numericas + ["customer_state"]].isnull().sum())

for coluna in pistas_numericas:
    base[coluna] = base[coluna].fillna(base[coluna].median())

print("\nNulos DEPOIS do tratamento:", base[pistas_numericas].isnull().sum().sum())

# %% [markdown]
# ## Encoding: transformar o estado (texto) em números
# O modelo não entende "SP" ou "AL". O get_dummies cria uma coluna para
# cada estado, com 1 se o pedido é daquele estado e 0 se não é.

# %% Encoding do estado
estados = pd.get_dummies(base["customer_state"], prefix="estado", dtype=int)
print("Colunas criadas para os estados:", estados.shape[1])
print(estados.columns.tolist()[:5], "...")

# %% Montar a base final
base_final = pd.concat(
    [base[pistas_numericas], estados, base[["atrasou"]]], axis=1
)

print("\nBase final:", base_final.shape[0], "pedidos,",
      base_final.shape[1] - 1, "pistas + a coluna 'atrasou'")
print(f"Taxa de atraso na base final: {base_final['atrasou'].mean() * 100:.1f}%")

# %% [markdown]
# ## Uma espiada: o prazo prometido tem relação com o atraso?
# Dividimos os pedidos em 4 faixas de prazo e vemos a taxa de atraso em cada.

# %% Atraso por faixa de prazo prometido
faixas = pd.qcut(base_final["prazo_prometido_dias"], 4,
                 labels=["Prazo curto", "Médio-curto", "Médio-longo", "Prazo longo"])
print((base_final.groupby(faixas, observed=True)["atrasou"].mean() * 100).round(1))

# %% Salvar a base limpa para a Etapa 3
base_final.to_csv("base_modelo.csv", index=False)
print("\nArquivo base_modelo.csv salvo!")

# %% [markdown]
# ## Minhas observações da Etapa 2 
#
# 1. A base final tem 96.470 pedidos e 35 pistas (8 numéricas + 27 colunas
#    de estado criadas pelo encoding). Nenhum pedido se perdeu ao juntar as
#    tabelas, e a taxa de atraso se manteve em 6,8%.
#
# 2. Quase não havia nulos: 22 pedidos sem peso e 16 sem volume, menos de
#    0,03% da base. Eles foram preenchidos com a mediana, que não é puxada
#    por valores extremos.
#
# 3. O prazo prometido tem relação com o atraso. Pedidos com prazo longo
#    atrasaram menos (5,1%), porque têm folga: mesmo com algum problema no
#    caminho, ainda dá para entregar dentro do prometido. A faixa médio-curta
#    foi a que mais atrasou (8,3%).
