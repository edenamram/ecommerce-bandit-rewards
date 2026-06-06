# rl-ecommerce-reward-comparison

Comparing reward formulations for contextual bandit-based product recommendation on the RetailRocket e-commerce dataset.

## Research Question

> **Which reward formulation leads a contextual bandit to recommend products in a way that maximizes purchases in e-commerce?**

---

## Background

This project adapts the contextual bandit framework from:

> Semenov, A., Rysz, M., Pandey, G., & Xu, G. (2022). *Diversity in news recommendations using contextual bandits.* Expert Systems With Applications, 195, 116478.

The original paper applies LinUCB and DivLinUCB to news recommendation using the Yahoo! R6A dataset. We port the same algorithms to the e-commerce domain using the RetailRocket dataset, and systematically compare three different reward formulations to understand which best drives purchase behavior.

---

## Reward Functions

| ID | Name | Formula | Motivation |
|----|------|---------|------------|
| R1 | Binary | 1 if transaction, 0 otherwise | Direct purchase signal |
| R2 | Weighted | view=0.1, addtocart=0.5, transaction=1.0 | Funnel-aware, denser signal |
| R3 | Max-Stage | Reward based on deepest funnel stage reached per item per session | Captures full purchase intent |

---

## Algorithms

All algorithms are taken directly from Semenov et al. (2022) and Li et al. (2010):

| Algorithm | Source | Description |
|-----------|--------|-------------|
| Random | Baseline | Uniform random arm selection |
| LinUCB | Li et al. (2010), used as baseline in Semenov et al. | Linear Upper Confidence Bound — learns which items maximize reward per user context |
| DivLinUCB | Algorithm 1, Semenov et al. (2022) | Diversity-boosting LinUCB — penalises over-recommended items via a cost term |

**Key parameters (from the paper):**
- `alpha = 2.62` → corresponds to P=0.99, δ=0.01 confidence bound (Section 5.2)
- `beta = 5000` → diversity strength (higher = more diverse recommendations)
- `n = 2` → exponent on the normalised recommendation count (Eq. 4)
- Gini Index → diversity metric (Eq. 6)

---

## Key Findings

| Reward | Best Algorithm | Why |
|--------|---------------|-----|
| R1 Binary | Random | Signal too sparse (1.35% purchase rate) — LinUCB cannot learn effectively |
| R2 Weighted | Random (marginal) | Denser signal but still noisy; algorithms concentrate on wrong items |
| **R3 Max-Stage** | **LinUCB** | Richest signal — rewards full session intent, allowing meaningful learning |

**Main conclusion:** R3 Max-Stage is the best reward formulation for e-commerce bandits. It provides a dense enough learning signal by rewarding the deepest funnel stage reached, rather than waiting for rare purchase events.

**Diversity tradeoff:** DivLinUCB achieves lower Gini than LinUCB on R3 (0.853 vs 0.841), confirming the reward–diversity tradeoff reported in Semenov et al. — more diverse recommendations come at a small cost to purchase rate.

---

## Dataset

RetailRocket E-Commerce Dataset (Kaggle)  
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Three event types in the log:
- `view` — user viewed a product
- `addtocart` — user added to cart
- `transaction` — user purchased

**Dataset limitation:** RetailRocket is an implicit feedback dataset with no explicit candidate pool per event (unlike Yahoo! R6A used in the original paper). We therefore use a warm-up / evaluation split:
- **Warm-up (50%):** All algorithms learn from the historical log
- **Evaluation (50%):** Each algorithm makes real independent choices; metrics are computed on the chosen arm using its historical mean reward

---

## Project Structure

```
rl-ecommerce-reward-comparison/
├── main.py                        # Run full experiment + save all plots
├── download_data.py               # Auto-download RetailRocket from Kaggle
├── requirements.txt
├── src/
│   └── preprocessing.py          # Load & prepare dataset, build PCA context vectors
├── algorithms/
│   ├── __init__.py
│   ├── random_bandit.py          # Random baseline
│   ├── linucb.py                 # LinUCB — Li et al. (2010)
│   └── divlinucb.py              # DivLinUCB — Semenov et al. (2022), Algorithm 1
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # CTR, Gini Index, purchase rate, reward curves
│   └── simulator.py              # Warm-up + evaluation offline loop
├── rewards/
│   ├── __init__.py
│   └── reward_functions.py       # R1, R2, R3 definitions
├── data/
│   └── raw/                      # CSV files land here after download
└── results/                      # All plots and summary_table.csv saved here
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourname/rl-ecommerce-reward-comparison.git
cd rl-ecommerce-reward-comparison
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Kaggle API credentials**

Go to https://www.kaggle.com/settings → API → Create New Token → download `kaggle.json`

```bash
# Mac / Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**5. Run**
```bash
python main.py
```

The dataset downloads automatically on first run. Results and plots are saved to `results/`.

---

## Output

| File | Description |
|------|-------------|
| `results/summary_table.csv` | Mean reward, purchase rate, Gini, match rate for all 9 combos |
| `results/ctr_comparison.png` | Mean reward per algorithm and reward function |
| `results/gini_comparison.png` | Gini index — recommendation diversity |
| `results/purchase_rate.png` | Purchase rate of chosen items |
| `results/learning_curves.png` | Average reward over time (evaluation phase) |
| `results/cumulative_rewards.png` | Cumulative reward over time |
| `results/recommendation_hist.png` | Item recommendation distribution per algorithm |

---

## References

- Semenov, A., Rysz, M., Pandey, G., & Xu, G. (2022). Diversity in news recommendations using contextual bandits. *Expert Systems With Applications*, 195, 116478.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW 2010*, 661–670.
- RetailRocket E-Commerce Dataset: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset# ecommerce-bandit-rewards
