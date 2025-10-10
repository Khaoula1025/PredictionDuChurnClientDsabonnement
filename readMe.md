# 📊 Prédiction du Churn Client - Télécommunications

> Modèle d'intelligence artificielle pour identifier les clients à risque de désabonnement et optimiser les stratégies de fidélisation.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Objectifs du Projet

Ce projet implémente un pipeline complet de Machine Learning supervisé visant à :

- ✅ Prédire le churn client avec une haute précision
- ✅ Comparer plusieurs algorithmes de classification
- ✅ Identifier les facteurs clés influençant le désabonnement
- ✅ Fournir des insights actionnables pour les équipes marketing
- ✅ Garantir la qualité et la reproductibilité du code

---

## 📁 Structure du Projet

```
predictionDuChurnClientDsabonnement/
│
├── 📄 data.csv                      # Dataset des clients
├── 📓 analyzeAndPrepareData.ipynb   # Analyse exploratoire (EDA)
├── 🐍 pipeline.py                   # Pipeline ML complet
├── 🧪 test_pipeline.py              # Tests unitaires
├── 📋 requirements.txt              # Dépendances Python
└── 📖 README.md                     # Documentation
```

---

## 🚀 Installation & Configuration

### Prérequis

- Python 3.8 ou supérieur
- pip ou conda

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Khaoula1025/PredictionDuChurnClientDsabonnement
cd telecom-churn-prediction

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Sur Linux/Mac :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## 💻 Utilisation

### Analyse Exploratoire des Données

```bash
jupyter notebook analyzeAndPrepareData.ipynb
```

### Exécution du Pipeline Complet

```bash
python pipeline.py
```

### Lancement des Tests

```bash
pytest test_pipeline.py
```

---

## 🔍 Analyse Exploratoire des Données (EDA)

### Problèmes Identifiés

| Problème | Solution Appliquée |
|----------|-------------------|
| **Type de données incorrect** | Conversion de `TotalCharges` en type numérique |
| **Variables catégorielles** | Encodage via `OneHotEncoder` |
| **Déséquilibre des classes** | Churn : ~26% (à surveiller) |
| **Valeurs manquantes** | Imputation ou suppression selon le contexte |


---

## ⚙️ Pipeline de Traitement des Données

Le pipeline automatise les étapes suivantes :

```
1. Nettoyage       → Suppression des doublons et valeurs aberrantes
2. Traitement      → Gestion des valeurs manquantes
3. Encodage        → OneHotEncoder pour variables catégorielles
4. Normalisation   → StandardScaler pour variables numériques
5. Séparation      → Train/Test split (80/20, random_state=42)
```

---

## 🤖 Modèles Implémentés

Deux algorithmes de classification ont été comparés :

### 1. **Logistic Regression**
- Modèle linéaire interprétable
- Rapide à entraîner
- Adapté aux relations linéaires

### 2. **Random Forest**
- Modèle ensembliste non-linéaire
- Capture les interactions complexes
- Robuste au surapprentissage

---

## 📊 Résultats et Performance

### Métriques de Comparaison

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| **Logistic Regression** ⭐ | **0.80** | **0.66** | **0.56** | **0.60** | **0.84** |
| Random Forest | 0.79 | 0.62 | 0.51 | 0.56 | 0.82 |

### 🏆 Modèle Sélectionné : **Logistic Regression**

**Justification :**
- ✅ Meilleur équilibre entre toutes les métriques
- ✅ Recall supérieur (crucial pour minimiser les faux négatifs)
- ✅ AUC plus élevée (meilleure discrimination)
- ✅ Plus interprétable pour l'équipe métier

### Métriques Prioritaires

- **Recall (0.56)** : Capture 56% des clients à risque réel
- **AUC (0.84)** : Excellente capacité de discrimination
- **F1-Score (0.60)** : Bon compromis précision/rappel

---

## 🧪 Tests et Validation
Un test automatisé été mis en place pour garantir la fiabilité du pipeline :

Vérification des dimensions cohérentes entre X et y après split .

### Exécution des Tests

```bash
pytest test_pipeline.py -v
```

---

## 📅 Planning du Projet

| Jour | Phase | Livrables |
|------|-------|-----------|
| **J1** | 🔍 Exploration | Analyse EDA, visualisations |
| **J2** | 🛠️ Préparation | Pipeline de nettoyage et preprocessing |
| **J3** | 🤖 Modélisation | Entraînement des modèles |
| **J4** | 📊 Évaluation | Métriques, comparaisons, sélection |
| **J5** | 📝 Documentation | README, tests, rapport final |

**Gestion de projet** : Suivi via Jira avec méthodologie Agile

---

## 🛠️ Technologies Utilisées

- **Python 3.8+** - Langage principal
- **Pandas** - Manipulation de données
- **NumPy** - Calcul numérique
- **Scikit-learn** - Machine Learning
- **Matplotlib/Seaborn** - Visualisations
- **Jupyter Notebook** - Analyse interactive
- **Pytest** - Tests unitaires

---


## 👤 Auteur

**Khaoula Esioudi**  
*Data Scientist Junior*



</div>






