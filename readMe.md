# 📊 Rapport Technique – Prédiction du Churn Client

## 📋 Description du Projet

Ce projet vise à développer un modèle d'intelligence artificielle pour prédire le churn (désabonnement) des clients d'une entreprise de télécommunications. Il s'agit d'un pipeline complet de Machine Learning supervisé permettant d'identifier les clients à risque de départ afin de mettre en place des campagnes de fidélisation ciblées.

---

## 🎯 Objectifs

* Construire un pipeline ML end-to-end pour la prédiction du churn
* Comparer plusieurs algorithmes de classification
* Fournir des métriques d'évaluation détaillées
* Garantir la qualité du code avec des tests unitaires
* Documenter le processus de sélection du modèle optimal

## 🗂️ Structure du Projet

```bash
telecom-churn-prediction/
│
├── data.csv                        # Dataset des clients
│
├── analyzeAndPrepareData.ipynb     # Exploration et analyses visuelles
│
├── pipeline.py                     # Pipeline de préparation et modélisation
│
├── test_pipeline.py                # Tests unitaires
│
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier


## 🧠 Exploration des Données (EDA)

L’analyse exploratoire a permis d’identifier :

* **Problèmes détectés** :

  * Type de données incorrect — la colonne TotalCharges est enregistrée comme objet (chaîne de caractères) alors qu’elle contient en réalité des valeurs numériques.
  * Variables catégorielles nécessitant encodage (`gender`, `InternetService`, etc.)
* **Déséquilibre de classes** : proportion de churners environ 26%.

Des visualisations ont été produites :

* Distribution du churn par type de contrat
* Corrélation entre variables numériques

---

## ⚙️ Pipeline de Préparation (pipeline.py)

### Étapes principales :

1. **Nettoyage des données** : suppression des doublons et traitement des valeurs manquantes
2. **Encodage** : utilisation de `OneHotEncoder` pour les variables catégorielles
3. **Normalisation** : `StandardScaler` sur les variables numériques
4. **Split des données** : `train_test_split` avec `test_size=0.2`, `random_state=42`

---

## 🤖 Modélisation

Deux modèles ont été testés :

| Modèle                  | 
| ----------------------- | 
| **Logistic Regression** |            
| **Random Forest**       |

---

## 📈 Évaluation et Résultats

Les performances ont été évaluées à l’aide des métriques suivantes :

* **Accuracy**
* **Recall** (priorité : minimiser les faux négatifs)
* **F1-score**
* **Courbe ROC**

| Modèle              | Accuracy | Recall   | F1-score | precision|    AUC   |
| ------------------- | -------- | -------- | -------- | -------- | -------- |
| Logistic Regression |    0.80  |   0.56   | 0.60     | 0.66     |   0.84   |   
|---------------------| -------- | -------- | -------- | -------- | -------- |
| **Random Forest**   |    0.79  |   0.51   | 0.56     | 0.62     |   0.82   |

**Conclusion :** Le modèle Logistic Regression offre le meilleur équilibre entre rappel et performance globale

---

## 🧪 Tests Unitaires (test_pipeline.py)

Un test automatisé été mis en place pour garantir la fiabilité du pipeline :

* Vérification des dimensions cohérentes entre `X` et `y` après split .

---

## 📅 Gestion du Projet (via Jira)

Le projet a été découpé en 5 jours de travail :

| Jour | Tâche principale        | Détails                                |
| ---- | ----------------------- | -------------------------------------- |
| 1    | EDA                     | Analyse et visualisation des données   |
| 2    | Préparation des données | Nettoyage, encodage, normalisation     |
| 3    | Modélisation            | Entraînement des 3 modèles             |
| 4    | Évaluation              | Comparaison, métriques, visualisations |
| 5    | Rapport final           | Rédaction du README, tests unitaires   |

---

## 👤 Auteur

**Khaoula Esioudi**
Data Scientist Junior – Projet de prédiction du churn client
2025




