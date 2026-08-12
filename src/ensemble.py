"""
Phase 7: Ensemble Methods
Voting, Stacking, and Bagging on top of the base classifiers.
"""
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_voting_ensemble(trained_models: dict) -> VotingClassifier:
    """Soft-voting ensemble across already-defined model types (re-fit fresh copies)."""
    estimators = [(name, model) for name, model in trained_models.items()]
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return voting


def build_stacking_ensemble(base_models: dict) -> StackingClassifier:
    """Stacking with a Logistic Regression meta-learner."""
    estimators = [(name, model) for name, model in base_models.items()]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1
    )
    return stacking


def build_bagging_ensemble(base_estimator=None, n_estimators: int = 50) -> BaggingClassifier:
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(max_depth=8, class_weight="balanced")
    return BaggingClassifier(
        estimator=base_estimator, n_estimators=n_estimators,
        random_state=42, n_jobs=-1
    )


def fit_all_ensembles(base_models: dict, X_train, y_train) -> dict:
    """Fit voting, stacking, and bagging ensembles; return trained instances."""
    # Voting/Stacking need un-fit estimators, so we clone via get_params re-instantiation
    from sklearn.base import clone
    fresh_models = {name: clone(m) for name, m in base_models.items()}

    voting = build_voting_ensemble(fresh_models)
    voting.fit(X_train, y_train)

    fresh_models_2 = {name: clone(m) for name, m in base_models.items()}
    stacking = build_stacking_ensemble(fresh_models_2)
    stacking.fit(X_train, y_train)

    bagging = build_bagging_ensemble()
    bagging.fit(X_train, y_train)

    return {"Voting": voting, "Stacking": stacking, "Bagging": bagging}
