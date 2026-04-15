# Detecção de Fraude em Transações Pix

**Disciplina:** SCC-276 - Aprendizado de Máquina (ICMC/USP)

---

## 1. Apresentação da Base de Dados

O dataset utilizado neste projeto é o [**pix-fraud-challenge-v1**](https://www.kaggle.com/datasets/raphaelnunes/pix-fraud-challenge-v1), disponível publicamente no Kaggle sob licença MIT (Raphael Nunes, 2026).

### Visão geral

| Característica | Valor |
|---|---|
| Nº de observações | 75.375 |
| Nº de atributos | 13 |
| Variável alvo | `is_fraud` (0 = legítima, 1 = fraude) |
| Taxa de fraude | ~3,05 % (2.296 fraudes) |
| Período coberto | Janeiro/2023 – Dezembro/2024 |
| Organizações | 10 (com perfis de risco distintos) |

O dataset é **sintético**, gerado para simular transações Pix reais com padrões de fraude realistas, o que o torna ideal para experimentação acadêmica sem riscos de privacidade.

### Descrição dos atributos

| Atributo | Tipo | Descrição |
|---|---|---|
| `transaction_id` | Categórico | Identificador único da transação (UUID) |
| `org_id` | Categórico | Identificador da organização financeira |
| `account_id` | Numérico | Identificador numérico da conta |
| `transaction_datetime` | Temporal | Data e hora da transação |
| `transaction_amount` | Numérico (R$) | Valor da transação em reais |
| `pix_key_type` | Categórico | Tipo de chave Pix utilizada (`cpf`, `email`, `telefone`, `aleatoria`) |
| `account_age_days` | Numérico | Idade da conta em dias |
| `account_state` | Categórico | Estado brasileiro do titular da conta |
| `account_device` | Categórico | Dispositivo utilizado (`mobile`, `desktop`, `tablet`) |
| `account_transactions_last_24h` | Numérico | Número de transações da conta nas últimas 24 horas |
| `account_avg_transaction_amount` | Numérico (R$) | Valor médio histórico das transações da conta |
| `internal_code` | Categórico | Código interno do sistema |
| `is_fraud` | Binário | **Variável alvo** — 0 (legítima) ou 1 (fraude) |

### Qualidade dos dados

A base apresenta valores ausentes em duas colunas — `account_state` (358 NaN) e `account_device` (3.717 NaN) — que deverão ser tratados durante o pré-processamento (e.g., imputação pela moda). Além disso, a distribuição das classes é **fortemente desbalanceada** (~97 % legítimas vs. ~3 % fraudes), o que exigirá técnicas específicas para evitar viés no treinamento, como oversampling com SMOTE.

---

## 2. Problema e Contextualização

### O problema

O **Pix**, sistema de pagamentos instantâneos criado pelo Banco Central do Brasil em 2020, ultrapassou 45 bilhões de transações anuais em 2025, consolidando-se como o principal meio de pagamento do país. Entretanto, a velocidade e a praticidade do sistema também o tornaram alvo de fraudadores — em 2024, fraudes via Pix geraram **R$ 4,9 bilhões** em perdas, motivando a criação do **MED (Mecanismo Especial de Devolução)** pelo Banco Central.

O objetivo deste projeto é **construir um modelo de aprendizado de máquina capaz de classificar automaticamente transações Pix como legítimas ou fraudulentas**, auxiliando a detecção precoce de fraudes e, potencialmente, alimentando sistemas de alerta como o MED.

### Tipo de problema: Classificação Binária Supervisionada

Dentro do escopo da disciplina de Aprendizado de Máquina, este projeto enquadra-se como um problema de **classificação binária supervisionada**: dado um conjunto de atributos descritivos de uma transação Pix, o objetivo é predizer o rótulo discreto `is_fraud ∈ {0, 1}`. A variável alvo já está disponível no dataset (aprendizado supervisionado) e possui exatamente duas classes (classificação binária).

### Modelos a serem investigados

Pretendemos comparar quatro algoritmos representativos de diferentes paradigmas:

| Modelo | Paradigma |
|---|---|
| **Regressão Logística** | Baseline linear |
| **Random Forest** | Ensemble - Bagging |
| **XGBoost** | Ensemble - Boosting |
| **MLP (Multilayer Perceptron)** | Rede neural artificial |

### Métricas de avaliação

Dada a natureza desbalanceada do problema, a **Accuracy** isolada é insuficiente (um classificador trivial que prediz sempre "legítima" já atingiria ~97 %). Portanto, pretendemos utilizar:

- **Recall** (sensibilidade) — métrica mais crítica: cada fraude não detectada representa prejuízo financeiro direto.
- **Precision** — proporção de alertas verdadeiramente fraudulentos.
- **F1-Score** — média harmônica entre Precision e Recall, equilibrando ambas.
- **AUC-ROC** — capacidade discriminativa ao longo de todos os thresholds de decisão.

A avaliação será conduzida com **Stratified 5-Fold Cross-Validation** e busca de hiperparâmetros para garantir robustez e reprodutibilidade.

### Engenharia de features (planejada)

A partir dos atributos brutos, planejamos criar variáveis derivadas específicas do domínio Pix, como:

- **Hora da transação** e **dia da semana** (extraídos de `transaction_datetime`)
- **Razão valor/média** (`transaction_amount / account_avg_transaction_amount`) — indicador de desvio comportamental
- **Indicador de madrugada** e **indicador de fim de semana** — padrões horários possivelmente associados a fraudes

A seleção final de features será feita com técnicas como **Mutual Information** para identificar os atributos mais discriminativos.

---

## Estrutura do Projeto

```
ML-pix-fraud/
├── projeto_deteccao_fraude_pix.ipynb   # Notebook principal (em desenvolvimento)
├── README.md                           # Este arquivo
├── data/
│   └── pix_fraud_v1.csv               # Dataset (75.375 transações)
└── figures/                            # Gráficos (a serem gerados)
```

## Como Executar

```bash
# Clonar o repositório
git clone <url-do-repositório>
cd ML-pix-fraud

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Abrir o notebook
jupyter notebook projeto_deteccao_fraude_pix.ipynb
```

## Referências

1. Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A Financial Mobile Money Simulator for Fraud Detection.* EMSS.
2. Carcillo, F. et al. (2021). *Combining unsupervised and supervised learning in credit card fraud detection.* Information Sciences, 557.
3. Hilal, W. et al. (2022). *Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances.* Expert Systems with Applications, 193.
4. Cherif, A. et al. (2023). *Credit card fraud detection in the era of disruptive technologies.* Journal of King Saud University.
5. Santos, R. M., & Silva, L. A. (2024). *Machine learning para detecção de fraudes no Pix.* Anais do SBSI.
6. Nunes, R. (2026). *pix-fraud-challenge-v1* [Dataset]. [Kaggle](https://www.kaggle.com/datasets/raphaelnunes/pix-fraud-challenge-v1).
