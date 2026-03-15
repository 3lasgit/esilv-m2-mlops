# 📹 Rapport & Vidéo Démo

## Lien vidéo démo

> **Vidéo de démonstration** (pipeline CI/CD + démo API) :
>
> 🔗 [https://github.com/3lasgit/esilv-m2-mlops](https://github.com/3lasgit/esilv-m2-mlops)

La vidéo couvre :

1. **Pipeline CI** — Exécution automatique des linters (ruff, black, isort, mypy) et des tests (pytest ≥ 60 % coverage) sur chaque push/PR
2. **Pipeline CD** — Build et push des images Docker (training + inference) vers GHCR lors du merge sur `main`, suivi d'un smoke test `/health`
3. **Démo API** — Lancement de l'API FastAPI (`/predict`, `/health`, `/metrics`), envoi d'un payload patient et interprétation de la réponse
4. **MLflow** — Visualisation du tracking des expériences, comparaison des runs, et model registry

---

## Structure des livrables

| Fichier | Description |
|---------|-------------|
| `README.md` (racine) | Documentation complète du projet |
| `reports/README.md` | Ce fichier — lien vidéo + résumé des livrables |
| `.github/workflows/ci.yml` | Pipeline CI (lint + tests) |
| `.github/workflows/cd.yml` | Pipeline CD (build Docker + smoke test) |
