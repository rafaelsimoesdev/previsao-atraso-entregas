# Projeto 2 — Etapa 3: Treinar os modelos de classificação
# Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
# Arquivo esperado (na mesma pasta deste arquivo):
#   - base_modelo.csv  (gerado na Etapa 2)
#
# OBJETIVO: treinar modelos que preveem se um pedido vai atrasar,
# comparando com um baseline e medindo com precisão e recall.

# %% [markdown]
# # Projeto 2 — Etapa 3: Os modelos
#
# ## Por que não usar só a acurácia?
# Apenas 6,8% dos pedidos atrasam. Um modelo que responde "não vai atrasar"
# para TODOS os pedidos acerta 93,2% das vezes — mas é inútil: ele nunca
# avisa nenhum atraso. Por isso usamos:
#
# - **Precisão:** dos pedidos que o modelo disse "vai atrasar", quantos
#   de fato atrasaram? (Evita alarmes falsos)
# - **Recall:** dos pedidos que realmente atrasaram, quantos o modelo
#   conseguiu identificar? (Evita deixar atrasos passarem)
#
# Para operação logística, o **recall** é mais importante: é melhor
# investigar um pedido que não atrasaria do que deixar um atraso real
# passar sem aviso.

# %% Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score)

# %% Carregar a base limpa
base = pd.read_csv("base_modelo.csv")
print(f"Base carregada: {base.shape[0]} pedidos, {base.shape[1]-1} pistas")
print(f"Taxa de atraso: {base['atrasou'].mean()*100:.1f}%")

# %% Separar pistas (X) da resposta (y)
X = base.drop(columns=["atrasou"])
y = base["atrasou"]

# %% Separar treino e teste (80% treino, 20% teste)
# random_state=42 garante que o resultado é o mesmo toda vez que rodamos
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {len(X_treino)} pedidos")
print(f"Teste:  {len(X_teste)} pedidos")
print(f"Atrasos no teste: {y_teste.sum()} ({y_teste.mean()*100:.1f}%)")

# %% [markdown]
# ## Baseline: sempre diz "não vai atrasar"
# Este é o modelo mais burro possível. Ele serve de piso mínimo:
# qualquer modelo real precisa ser melhor que isso.

# %% Baseline
y_baseline = pd.Series([0] * len(y_teste), index=y_teste.index)

prec_base = precision_score(y_teste, y_baseline, zero_division=0)
rec_base  = recall_score(y_teste, y_baseline, zero_division=0)
acc_base  = (y_baseline == y_teste).mean()

print("\n--- BASELINE (sempre 'no prazo') ---")
print(f"Acurácia:  {acc_base*100:.1f}%")
print(f"Precisão:  {prec_base*100:.1f}%")
print(f"Recall:    {rec_base*100:.1f}%")
print("→ Acurácia alta, mas recall 0%: nunca detecta nenhum atraso.")

# %% [markdown]
# ## Random Forest
# Conjunto de várias árvores de decisão votando juntas.
# max_depth limita o crescimento de cada árvore para evitar overfitting.
# class_weight='balanced' diz ao modelo para prestar mais atenção
# nos atrasos (que são raros) — essencial para dados desbalanceados.

# %% Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_treino, y_treino)
y_pred_rf = rf.predict(X_teste)

prec_rf = precision_score(y_teste, y_pred_rf)
rec_rf  = recall_score(y_teste, y_pred_rf)
acc_rf  = (y_pred_rf == y_teste).mean()

print("\n--- RANDOM FOREST ---")
print(f"Acurácia:  {acc_rf*100:.1f}%")
print(f"Precisão:  {prec_rf*100:.1f}%")
print(f"Recall:    {rec_rf*100:.1f}%")

# %% [markdown]
# ## Gradient Boosting
# Constrói árvores em sequência: cada nova árvore tenta corrigir os
# erros da anterior. Costuma ter performance melhor que RF, mas é mais
# lento para treinar.

# %% Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_treino, y_treino)
y_pred_gb = gb.predict(X_teste)

prec_gb = precision_score(y_teste, y_pred_gb)
rec_gb  = recall_score(y_teste, y_pred_gb)
acc_gb  = (y_pred_gb == y_teste).mean()

print("\n--- GRADIENT BOOSTING ---")
print(f"Acurácia:  {acc_gb*100:.1f}%")
print(f"Precisão:  {prec_gb*100:.1f}%")
print(f"Recall:    {rec_gb*100:.1f}%")

# %% [markdown]
# ## Comparação final

# %% Tabela comparativa
print("\n" + "="*55)
print(f"{'Modelo':<22} {'Acurácia':>9} {'Precisão':>9} {'Recall':>9}")
print("-"*55)
print(f"{'Baseline':<22} {acc_base*100:>8.1f}% {prec_base*100:>8.1f}% {rec_base*100:>8.1f}%")
print(f"{'Random Forest':<22} {acc_rf*100:>8.1f}% {prec_rf*100:>8.1f}% {rec_rf*100:>8.1f}%")
print(f"{'Gradient Boosting':<22} {acc_gb*100:>8.1f}% {prec_gb*100:>8.1f}% {rec_gb*100:>8.1f}%")
print("="*55)

# %% [markdown]
# ## Matriz de confusão do melhor modelo
# Mostra exatamente o que o modelo acertou e errou:
#
#                   Previu: No prazo  |  Previu: Atrasou
#   Real: No prazo       TN (certo)   |   FP (alarme falso)
#   Real: Atrasou        FN (perdeu)  |   TP (acertou atraso)
#
# Para logística, o pior erro é o FN: pedido que atrasou mas o modelo
# disse que não iria atrasar.

# %% Matriz de confusão do Random Forest
print("\n--- Matriz de confusão: Random Forest ---")
cm_rf = confusion_matrix(y_teste, y_pred_rf)
print(f"                Previu: No prazo  Previu: Atrasou")
print(f"Real: No prazo       {cm_rf[0,0]:>6}           {cm_rf[0,1]:>6}")
print(f"Real: Atrasou        {cm_rf[1,0]:>6}           {cm_rf[1,1]:>6}")

# %% Matriz de confusão do Gradient Boosting
print("\n--- Matriz de confusão: Gradient Boosting ---")
cm_gb = confusion_matrix(y_teste, y_pred_gb)
print(f"                Previu: No prazo  Previu: Atrasou")
print(f"Real: No prazo       {cm_gb[0,0]:>6}           {cm_gb[0,1]:>6}")
print(f"Real: Atrasou        {cm_gb[1,0]:>6}           {cm_gb[1,1]:>6}")

# %% [markdown]
# ## Pistas mais importantes (Random Forest)
# Quais informações o modelo mais usou para decidir?

# %% Feature importance
importancias = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importancias.sort_values(ascending=False).head(10)

print("\n--- Top 10 pistas mais importantes (Random Forest) ---")
for pista, valor in top10.items():
    barra = "█" * int(valor * 200)
    print(f"{pista:<30} {valor:.3f}  {barra}")

# %% [markdown]
# ## Minhas observações da Etapa 3
#
# 1. O baseline tinha recall 0% porque ele sempre responde "não vai atrasar"
#    para todos os pedidos. Com apenas 6,8% de atrasos, essa estratégia
#    acerta 93,2% das vezes — mas nunca detecta nenhum atraso real.
#    É um modelo inútil para a operação logística.
#
# 2. O Random Forest teve o melhor recall: 61%. Isso significa que ele
#    identifica 6 de cada 10 pedidos que vão atrasar antes da entrega.
#    Para operação logística isso é muito valioso: permite agir com
#    antecedência nos pedidos de risco. O Gradient Boosting teve recall
#    de apenas 2,8% — quase não detectou nenhum atraso.
#
# 3. As pistas mais importantes foram:
#    - mes_compra (0.327): o mês da compra é de longe o fator mais
#      relevante. Meses de pico como novembro (Black Friday) e início
#      de 2018 tinham taxas de atraso muito maiores.
#    - prazo_prometido_dias (0.165): prazos intermediários atrasam mais
#      que prazos curtos ou longos — confirmado na análise da Etapa 2.
#    - estado_SP e estado_RJ: os dois maiores destinos do país aparecem
#      como pistas relevantes, refletindo o volume de pedidos.
#    - valor_frete (0.091): fretes mais caros indicam destinos mais
#      distantes, que tendem a atrasar mais.
