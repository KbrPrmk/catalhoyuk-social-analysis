# 🏛️ Data Analysis of Social Structure at Çatalhöyük
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-darkgreen)
![SVM](https://img.shields.io/badge/Model-SVM-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Research](https://img.shields.io/badge/Domain-Archaeology%20%2B%20ML-purple)

A machine learning study testing social equality hypotheses at **Çatalhöyük** (7100–5950 BCE) — one of the world's earliest and most densely populated Neolithic settlements. Using archaeological, paleogenomic, and isotopic datasets, this project investigates whether this ancient community was truly egalitarian or whether hidden hierarchies shaped its social fabric.

---

## 📌 Research Questions

1. **Elite Houses** — When clustered by architectural complexity, room count, hearths, and storage features, do "elite houses" emerge that indicate class-based distinctions?
2. **Grave Goods by Sex** — Is the distribution of mortuary offerings (quantity and type) statistically different between male and female individuals?
3. **Skeletal Trauma** — What does the distribution of cranial trauma patterns reveal about violence and conflict resolution in this society?

---

## 📂 Data Sources

| File | Contents |
|------|----------|
| `catalhoyuk_buildings_improved_1_300.csv` | 137 building records: architectural complexity scores, room/hearth/pit counts, burial density |
| `catalhoyuk_skeleton_units_1000_3000.csv` | 81 skeletal units: individual IDs, trauma scores, burial context |
| `science.adr2915_data_s1_to_s28.xlsx` | Paleogenomic & isotopic data — Sheets S1 (sampling), S18 (diet isotopes), S21 (grave goods) |

> ⚠️ **Note:** `science.adr2915_data_s1_to_s28.xlsx` is the open-access supplementary dataset from Yüncü et al. (2025), published in *Science*. It is not included in this repository.
> Access it here: Yüncü et al. (2025), *Science* 387, eadr2915. https://doi.org/10.1126/science.adr2915

The two CSV files were produced via **text mining** of Çatalhöyük Research Project (ÇRP) excavation archives, field notes, and published inventories.

---

## 🔬 Methodology
```bash
Raw Data (3 sources)
    │
    ├── Per-dataset cleaning (isnull, dropna, fillna, mode imputation)
    ├── Outlier detection (Boxplot + IQR) — preserved as social evidence
    │
    ├── Inner + Left Joins → 104 individuals × 66 variables
    │       Sample ID / Individual / Building ID / Unit ID
    │
    ├── Feature Engineering
    │       └── house_class: Standard / Developed / Elite House
    │           (architectural_complexity_score bins: 0–200 / 200–400 / 400+)
    │
    ├── Machine Learning (GridSearchCV, 5-fold CV)
    │       ├── Q1 — House Classification: RF / XGBoost / SVM
    │       └── Q2 — Any Gift Prediction:  RF / XGBoost / SVM
    │
    └── Statistical Tests
            ├── Friedman test (model comparison)
            ├── Mann-Whitney U (grave goods × sex)
            └── Chi-square (gift ownership × sex)
```

---

## 📊 Libraries

```python
pandas       # Data wrangling and merging
numpy        # Numerical operations
seaborn      # Statistical visualizations
matplotlib   # Plotting
scikit-learn # ML models, GridSearchCV, metrics
xgboost      # XGBoost classifier
scipy        # Statistical tests
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn xgboost scipy openpyxl

# Launch the notebook
jupyter notebook catalhoyuk-social-analysis.ipynb
```

> Make sure all data files are in the same directory as the notebook.

---

## 📈 Key Findings

### Q1 — Do "Elite Houses" Exist?
XGBoost achieved the highest cross-validation F1-macro score (**0.910**) for house classification. Of 137 buildings, only **4 (2.9%)** were classified as Elite Houses — distinguished primarily by burial density, fire installation count, and room count. This aligns with Twiss et al.'s "history houses" concept: status was expressed through **ritualistic and symbolic capacity**, not wealth accumulation.

| House Class | n | % |
|-------------|---|---|
| Standard House | 76 | 73.1% |
| Developed House | 24 | 23.1% |
| Elite House | 4 | 3.8% |

The following table compares the three algorithms tested for house classification. The Friedman test (p = 0.607) confirmed all three models performed equivalently, meaning the pattern is data-driven, not algorithm-specific. XGBoost was selected as the final model.

| Model | Best Parameters | CV F1-Macro | Std Dev | Test Accuracy |
|-------|----------------|-------------|---------|---------------|
| Random Forest | n_estimators=100, max_depth=3 | 0.880 | ±0.134 | 95.2% |
| **XGBoost** | lr=0.05, max_depth=3 | **0.910** | ±0.141 | **95.2%** |
| SVM | C=10, kernel=rbf | 0.900 | ±0.136 | 90.5% |

---

### Q2 — Do Grave Goods Differ by Sex?
**52.0% of females** were interred with grave goods vs. **23.5% of males**. Across 7 of 8 object categories, females showed higher frequencies — shell and stone objects appeared exclusively in female burials. Notably, **100% of individuals in Elite Houses** were buried with grave goods.

| Gender | Beads | Shell | Blades | Wood Objects | Bone Objects | Stone Objects | Pouch Baskets | Pigments |
|--------|-------|-------|--------|--------------|--------------|---------------|---------------|----------|
| Female | 0.296 | 0.037 | 0.074 | 0.000 | 0.148 | 0.074 | 0.185 | 0.148 |
| Male | 0.053 | 0.000 | 0.053 | 0.000 | 0.053 | 0.000 | 0.105 | 0.105 |

Statistical tests did not reach significance (Mann-Whitney U: p = 0.066; χ²: p = 0.118), likely reflecting limited sample size (n = 46) rather than true independence. The table below shows model performance for predicting grave good ownership. Due to class imbalance (only 20.4% positive class), XGBoost failed to learn the minority class; Random Forest was selected as the final model.

| Model | CV ROC-AUC | Std Dev | Test Accuracy | F1 (Gift) | Precision (Gift) | Recall (Gift) |
|-------|------------|---------|---------------|-----------|------------------|---------------|
| **Random Forest** | **0.725** | ±0.120 | **75.0%** | 0.29 | 0.33 | 0.25 |
| XGBoost | —* | — | 75.0% | 0.29 | 0.33 | 0.25 |
| SVM | 0.661 | ±0.093 | 65.0% | 0.50 | 0.20 | 0.25 |

*\* XGBoost failed to learn the minority class during CV due to class imbalance (NaN scores).*

---

### Q3 — What Do Trauma Patterns Reveal?
Only 4 valid trauma records existed in the merged dataset — insufficient for statistical modeling. Of these, **3 came from Elite House contexts**, suggesting violence was not exclusively linked to deprivation but may reflect status competition. This remains exploratory due to sample size constraints.

---

### Correlation Analysis
The strongest correlation was between **Beads and Any Gift (r = 0.67)**, confirming beads as the primary symbolic status marker. Architectural complexity showed a **negative correlation with δ¹⁵N (r = −0.24)**, suggesting elite-house individuals did not consume higher-protein diets — consistent with Larsen et al.'s resource equality thesis and supporting the idea that status was ritualistic rather than material.

---

## 🏺 Conclusion

Çatalhöyük was neither fully egalitarian nor rigidly hierarchical. Social differentiation was shaped by **ritualistic practices, symbolic house organization, and likely matrilineal kinship structures** — not by wealth or resource hoarding. 

Key takeaways:
- Only 4 of 137 buildings (2.9%) qualify as Elite Houses, yet all individuals buried within them received grave goods
- Female individuals were interred with grave goods at more than twice the rate of males across nearly all object categories
- Status at Çatalhöyük was not accumulated through material wealth but crystallized through ritual, symbolism, and likely matrilineal household organization
- Diet isotope data (δ¹⁵N) shows no meaningful difference across house classes, suggesting resource access was broadly shared despite status distinctions

---

## 🔗 Key References

- Yüncü et al. (2025). *Science* 387, eadr2915. https://doi.org/10.1126/science.adr2915
- Twiss et al. (2024). *Cambridge Archaeological Journal* 34, 189–214.
- Schotsmans et al. (2025). *Nature Human Behaviour* 9, 78–94.
- Larsen et al. (2019). *PNAS* 116, 12615–12623.

---

## 📄 License

This project is shared under the MIT License. Users are responsible for complying with the individual licenses of each data source.
