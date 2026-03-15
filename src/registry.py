# ============================================================
# src/registry.py
# Gestion du MLflow Model Registry
# ============================================================
# Usage standalone :
#   python src/registry.py --promote          # promeut le meilleur run
#   python src/registry.py --compare          # compare les runs récents
#   python src/registry.py --promote --compare
# ============================================================

import argparse

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "heart-disease-prediction"
REGISTERED_MODEL_NAME = "heart-disease-classifier"
METRIC_KEY = "test_auc"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_client() -> MlflowClient:
    return MlflowClient()


def _get_experiment_id(client: MlflowClient) -> str:
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(
            f"Expérience '{EXPERIMENT_NAME}' introuvable. "
            "Lance d'abord `python src/train.py`."
        )
    return exp.experiment_id


# ------------------------------------------------------------------
# Comparaison des runs
# ------------------------------------------------------------------

def compare_runs(n: int = 10) -> None:
    """
    Affiche un tableau comparatif des n derniers runs de l'expérience,
    triés par test_auc décroissant.
    """
    client = _get_client()
    exp_id = _get_experiment_id(client)

    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="",
        order_by=[f"metrics.{METRIC_KEY} DESC"],
        max_results=n,
    )

    if not runs:
        print("Aucun run trouvé.")
        return

    print(f"\n{'=' * 72}")
    print(f"  COMPARAISON DES {min(n, len(runs))} DERNIERS RUNS — {EXPERIMENT_NAME}")
    print(f"{'=' * 72}")
    print(f"{'Run name':<28} {'AUC':>7} {'F1':>7} {'Acc':>7}  {'Run ID'}")
    print("-" * 72)

    for run in runs:
        name    = run.data.tags.get("mlflow.runName", "—")[:27]
        auc     = run.data.metrics.get("test_auc", float("nan"))
        f1      = run.data.metrics.get("test_f1",  float("nan"))
        acc     = run.data.metrics.get("test_accuracy", float("nan"))
        run_id  = run.info.run_id[:8]
        print(f"{name:<28} {auc:>7.4f} {f1:>7.4f} {acc:>7.4f}  {run_id}...")

    print(f"{'=' * 72}\n")


# ------------------------------------------------------------------
# Promotion du meilleur run
# ------------------------------------------------------------------

def promote_best_run() -> str:
    """
    Trouve le run avec le meilleur test_auc parmi les runs parents
    (non nested), enregistre le modèle dans le registry et le transite
    vers le stage 'Staging'.

    Returns
    -------
    run_id : str — identifiant du run promu
    """
    client = _get_client()
    exp_id = _get_experiment_id(client)

    # On cherche uniquement les runs parents (pas les nested child runs)
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="tags.mlflow.parentRunId = ''",
        order_by=[f"metrics.best_{METRIC_KEY} DESC"],
        max_results=1,
    )

    # Fallback : si le tag parentRunId n'est pas filtrable, prendre tous les runs
    if not runs:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=[f"metrics.best_{METRIC_KEY} DESC"],
            max_results=1,
        )

    if not runs:
        raise RuntimeError("Aucun run trouvé dans l'expérience.")

    best_run = runs[0]
    run_id   = best_run.info.run_id
    best_auc = best_run.data.metrics.get(f"best_{METRIC_KEY}", float("nan"))
    run_name = best_run.data.tags.get("mlflow.runName", run_id[:8])

    print(f"\n🏆 Meilleur run : '{run_name}' (id: {run_id[:8]}...) — best_test_auc = {best_auc:.4f}")

    # Récupérer l'URI du modèle RF tuné dans les child runs
    child_runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
        order_by=[f"metrics.{METRIC_KEY} DESC"],
        max_results=10,
    )

    model_uri = None
    for child in child_runs:
        tag = child.data.tags.get("model_type", "")
        if "RF" in tag or "Random Forest" in tag:
            model_uri = f"runs:/{child.info.run_id}/model"
            print(f"   Modèle sélectionné : RF (Tuned) — child run {child.info.run_id[:8]}...")
            break

    if model_uri is None:
        model_uri = f"runs:/{run_id}/model"
        print(f"   Modèle sélectionné : run parent {run_id[:8]}...")

    # Enregistrement dans le registry
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = model_details.version
    print(f"   Modèle enregistré — version {version} dans '{REGISTERED_MODEL_NAME}'")

    # Transition vers Staging
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"   Stage → Staging ✅")

    # Ajout d'une description sur la version
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            f"RF Tuned — test_auc={best_auc:.4f} | "
            f"run_id={run_id[:8]} | "
            f"experiment={EXPERIMENT_NAME}"
        ),
    )

    return run_id


def get_production_model_uri() -> str:
    """
    Retourne l'URI MLflow du modèle en Production.
    Utilisé par l'API FastAPI pour charger le modèle.

    Returns
    -------
    uri : str — ex. "models:/heart-disease-classifier/Production"
    """
    return f"models:/{REGISTERED_MODEL_NAME}/Production"


def promote_staging_to_production() -> None:
    """
    Transite la dernière version en Staging vers Production
    et archive l'ancienne version en Production.
    Appelé manuellement après validation.
    """
    client = _get_client()
    staging_versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME, stages=["Staging"]
    )
    if not staging_versions:
        print("Aucune version en Staging.")
        return

    v = staging_versions[0]
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=v.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"✅ Version {v.version} → Production | anciennes versions archivées.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Model Registry helper")
    parser.add_argument("--compare",  action="store_true", help="Afficher la comparaison des runs")
    parser.add_argument("--promote",  action="store_true", help="Promouvoir le meilleur run en Staging")
    parser.add_argument("--to-prod",  action="store_true", help="Passer le modèle Staging en Production")
    parser.add_argument("--n",        type=int, default=10, help="Nombre de runs à afficher (--compare)")
    args = parser.parse_args()

    if args.compare:
        compare_runs(n=args.n)
    if args.promote:
        promote_best_run()
    if args.to_prod:
        promote_staging_to_production()
    if not any([args.compare, args.promote, args.to_prod]):
        parser.print_help()