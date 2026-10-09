# Projeto 2 — Previsão de Atraso de Entrega

**Autor:** Rafael Simões  
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle  
**Período dos dados:** 2016 a 2018  
**Pergunta:** *Este pedido vai atrasar em relação à data prometida ao cliente?*

---

## Contexto

Atrasos de entrega são um dos maiores problemas de satisfação no e-commerce. Identificar com antecedência quais pedidos têm risco de atraso permite que a operação logística tome ações preventivas antes que o cliente seja impactado.

Este projeto treina um modelo de classificação capaz de prever, **no momento da compra**, se um pedido vai atrasar — usando apenas informações disponíveis naquele instante.

---

## Estrutura do Projeto

| Arquivo | Descrição |
|---|---|
| `p2_etapa1_explorar_atrasos.py` | Análise exploratória: taxa de atraso, padrões por mês e por estado |
| `p2_etapa2_preprocessamento.py` | Junção das tabelas, criação de pistas, tratamento de nulos e encoding |
| `p2_etapa3_modelos.py` | Treinamento e comparação dos modelos de classificação |
| `base_modelo.csv` | Base limpa gerada na Etapa 2 (não versionada) |

---

## Etapa 1 — Exploração dos Atrasos

**96.470 pedidos entregues** analisados entre 2016 e 2018.

- **Taxa de atraso geral: 6,8%** (6.534 pedidos)
- A taxa é baixa no geral, mas concentrada em regiões e períodos específicos

**Estados com maior taxa de atraso:**

| Estado | Taxa |
|---|---|
| Alagoas (AL) | 21,4% |
| Maranhão (MA) | 17,4% |
| Sergipe (SE) | 15,2% |
| Piauí (PI) | 13,9% |
| Ceará (CE) | 13,8% |

Os 5 estados com maior atraso são todos do Nordeste — reflexo da distância dos centros de distribuição concentrados no Sudeste e do maior número de etapas logísticas.

**Variação mensal:** a taxa de atraso saltou para 12% em novembro de 2017 (Black Friday), 14% em fevereiro de 2018 e 19% em março de 2018 — períodos de pico operacional que sobrecarregam a cadeia logística.

---

## Etapa 2 — Pré-processamento

**Regra de ouro:** só foram usadas informações disponíveis no momento da compra — sem vazamento de dados (data leakage).

**Pistas criadas (35 no total):**

| Pista | Descrição |
|---|---|
| `mes_compra` | Mês em que o pedido foi feito |
| `dia_semana_compra` | Dia da semana da compra |
| `prazo_prometido_dias` | Dias entre a compra e a data prometida |
| `qtd_itens` | Quantidade de itens no pedido |
| `valor_produtos` | Valor total dos produtos |
| `valor_frete` | Valor do frete |
| `peso_total_g` | Peso total dos produtos em gramas |
| `volume_total_cm3` | Volume total em cm³ |
| `estado_XX` (27 colunas) | Estado do cliente em formato one-hot encoding |

**Tratamento de nulos:** 22 pedidos sem peso e 16 sem volume (0,03% da base) — preenchidos com a mediana para não distorcer o modelo com valores extremos.

**Relação prazo × atraso:**

| Faixa de prazo | Taxa de atraso |
|---|---|
| Prazo curto | 6,3% |
| Médio-curto | 8,3% |
| Médio-longo | 7,3% |
| Prazo longo | 5,1% |

Prazos intermediários atrasam mais — provavelmente porque prazos muito curtos são dados apenas quando a entrega é garantida, e prazos muito longos têm folga suficiente para absorver imprevistos.

---

## Etapa 3 — Modelos de Classificação

### Por que não usar só a acurácia?

Com apenas 6,8% de atrasos, um modelo que responde "não vai atrasar" para todos os pedidos acerta **93,2%** das vezes — mas nunca detecta nenhum atraso. Por isso as métricas principais são:

- **Precisão:** dos pedidos que o modelo sinalizou como atraso, quantos realmente atrasaram?
- **Recall:** dos pedidos que realmente atrasaram, quantos o modelo conseguiu identificar?

Para operação logística, o **recall é mais importante**: é melhor investigar um pedido desnecessariamente do que deixar um atraso real passar sem aviso.

### Resultados

| Modelo | Acurácia | Precisão | Recall |
|---|---|---|---|
| Baseline (sempre "no prazo") | 93,2% | 0,0% | 0,0% |
| Gradient Boosting | 93,2% | 48,1% | 2,8% |
| **Random Forest** | **75,2%** | **15,7%** | **61,0%** |

### Matriz de confusão — Random Forest

|  | Previu: No prazo | Previu: Atrasou |
|---|---|---|
| **Real: No prazo** | 13.709 | 4.278 |
| **Real: Atrasou** | 510 | 797 |

O modelo identificou **797 dos 1.307 atrasos** no conjunto de teste — 61% de recall. Os 510 atrasos não detectados e os 4.278 alarmes falsos representam o trade-off esperado: para pegar mais atrasos, o modelo é mais conservador e sinaliza mais pedidos.

### Pistas mais importantes (Random Forest)

| Pista | Importância |
|---|---|
| mes_compra | 0,327 |
| prazo_prometido_dias | 0,165 |
| estado_SP | 0,097 |
| valor_frete | 0,091 |
| estado_RJ | 0,080 |
| valor_produtos | 0,040 |
| peso_total_g | 0,035 |
| volume_total_cm3 | 0,033 |
| estado_MG | 0,027 |
| estado_BA | 0,021 |

O **mês da compra** é de longe a pista mais importante — confirmando que períodos de pico (Black Friday, início de 2018) são o principal fator de risco. O **prazo prometido** é o segundo fator mais relevante.

---

## Conclusões

- O Random Forest com `class_weight='balanced'` foi o modelo mais útil para a operação, detectando 61% dos atrasos antes da entrega
- O Gradient Boosting teve precisão maior (48%) mas recall de apenas 2,8% — quase inútil para detectar atrasos na prática
- O mês da compra e o prazo prometido são os maiores sinalizadores de risco
- Os estados do Nordeste concentram as maiores taxas de atraso, reflexo das distâncias logísticas

---

## Como Reproduzir

```bash
# Instalar dependências
pip install pandas scikit-learn matplotlib

# Rodar as etapas em ordem
python p2_etapa1_explorar_atrasos.py
python p2_etapa2_preprocessamento.py
python p2_etapa3_modelos.py
```

Os arquivos CSV do dataset Olist precisam estar na mesma pasta dos scripts.  
Download: [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
