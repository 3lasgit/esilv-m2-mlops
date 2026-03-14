# 🫀 Heart Disease Prediction — MLOps Final Project

> **Cours** : Machine Learning & MLOps  
> **Encadrants** : yue.li@ext.devinci.fr · linghao.kong@ext.devinci.fr  
> **Deadline** : 15 mars 2026  

---

## Description

Ce projet applique un pipeline MLOps complet à la prédiction de maladies cardiaques à partir de données cliniques. L'objectif est de produire un modèle de classification binaire (présence ou absence de maladie cardiaque) reproductible, versionné, déployable et monitoré, en suivant les bonnes pratiques de l'ingénierie ML en production.

Le projet couvre l'ensemble de la chaîne : exploration des données, feature engineering, entraînement de modèles classiques et d'un réseau de neurones, experiment tracking avec MLflow, exposition via une API REST, containerisation Docker, et monitoring en production.

---

## Dataset — Heart Disease UCI (Cleveland)

| Propriété | Valeur |
|-----------|--------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Origine | Cleveland Clinic Foundation (1988) |
| Observations | 303 patients |
| Features | 13 variables cliniques |
| Cible | `target` — 0 = absence, 1 = présence de maladie cardiaque |
| Équilibre | ~54 % positifs / 46 % négatifs |

### Variables

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numérique | Âge du patient (années) |
| `sex` | Binaire | Sexe (1 = homme, 0 = femme) |
| `cp` | Catégorielle (0–3) | Type de douleur thoracique |
| `trestbps` | Numérique | Pression artérielle au repos (mmHg) |
| `chol` | Numérique | Cholestérol sérique (mg/dl) |
| `fbs` | Binaire | Glycémie à jeun > 120 mg/dl |
| `restecg` | Catégorielle (0–2) | Résultats ECG au repos |
| `thalach` | Numérique | Fréquence cardiaque maximale atteinte |
| `exang` | Binaire | Angine induite par l'effort |
| `oldpeak` | Numérique | Dépression du segment ST à l'effort |
| `slope` | Catégorielle (0–2) | Pente du segment ST |
| `ca` | Numérique (0–3) | Nombre de vaisseaux colorés (fluoroscopie) |
| `thal` | Catégorielle (0–2) | Thalassémie |

### Features dérivées (feature engineering)

Trois variables d'interaction à pertinence clinique ont été créées :

| Feature | Formule | Interprétation |
|---------|---------|----------------|
| `hr_age_ratio` | `thalach / age` | Réserve cardiaque normalisée par l'âge |
| `cardio_risk_score` | `(trestbps/120) × (chol/200) × (age/60)` | Score de risque cardiovasculaire composé |
| `exang_oldpeak` | `exang × oldpeak` | Stress cardiaque combiné |

---

## Task Definition

### Problème

**Classification binaire supervisée** : prédire si un patient présente une maladie cardiaque (`target = 1`) ou non (`target = 0`) à partir de ses données cliniques et biologiques.

### Métrique principale

**AUC-ROC** — choisie car elle est insensible au seuil de décision et adaptée aux applications médicales où l'équilibre précision/rappel est critique.  
Métriques secondaires : F1-score, Accuracy.

### Modèles entraînés

| Modèle | Type | Tuning |
|--------|------|--------|
| Logistic Regression | Linéaire | — |
| K-Nearest Neighbors | Instance-based | Sélection du k optimal (CV) |
| Decision Tree | Arbre | — |
| Random Forest | Ensemble | GridSearchCV |
| SVM (RBF) | Noyau | RandomizedSearchCV |
| ANN | Deep Learning | Early Stopping + ReduceLROnPlateau |

### Résultats clés (test set)

| Modèle | AUC-ROC | F1-Score | Accuracy |
|--------|---------|----------|----------|
| RF (Tuned) | **0.9012** | **0.8649** | — |
| SVM (Tuned) | 0.9032 | — | — |
| KNN | 0.8918 | — | — |
| ANN | ~0.84 | — | — |

> Les valeurs exactes sont loggées dans MLflow lors de chaque run d'entraînement (`python src/train.py`).

---

## Stack technique

| Composant | Outil |
|-----------|-------|
| Environnement | [uv](https://github.com/astral-sh/uv) |
| Qualité de code | ruff · black · isort · mypy · pre-commit |
| ML | scikit-learn · TensorFlow/Keras · SHAP |
| Experiment tracking | MLflow |
| API | FastAPI + Pydantic |
| Containerisation | Docker · docker-compose |
| CI/CD | GitHub Actions |
| Tests | pytest (≥ 60 % coverage) |

---

## Structure du projet

```
.
├── src/
│   ├── data.py          # Chargement, preprocessing, feature engineering
│   ├── model.py         # Définition des modèles, tuning, ANN
│   └── train.py         # Script d'entraînement principal
├── api/
│   └── app.py           # Endpoint FastAPI /predict
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── models/              # Artefacts sauvegardés (gitignore)
├── reports/             # Figures, rapport final, lien vidéo
├── .github/workflows/   # Pipelines CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml       # Dépendances gérées avec uv
└── README.md
```

---

## Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone <repo-url> && cd <repo>

# 2. Installer l'environnement avec uv
uv sync

# 3. Lancer l'entraînement
uv run python src/train.py

# 4. Lancer l'API (après entraînement)
uv run uvicorn api.app:app --reload
```

---

## Répartition des rôles

| Branche | Responsable | Contenu |
|---------|-------------|---------|
| `feature/project-structure` | A | Init uv, refactoring notebook → `src/`, README initial |
| `feature/code-quality` | B | Pre-commit hooks, tests unitaires (≥ 60 %), CI lint/pytest |
| `feature/mlflow` | A | Intégration MLflow (tracking + model registry) |
| `feature/api-docker` | B | FastAPI `/predict`, chargement depuis MLflow, Dockerfile |
| `feature/cicd-monitoring` | A | Pipeline CD, endpoint `/health`, logging structuré |
| `feature/final-report` | B | README final complet, lien vidéo démo |

Chaque branche fait l'objet d'une Pull Request revue par le binôme avant merge sur `main`.

---

## Auteurs

| Rôle | Contact |
|------|---------|
| Étudiant A | — |
| Étudiant B | — |