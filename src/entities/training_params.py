from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=255)
    n_estimators: int = field(default=100)
    n_neighbors: int = field(default=5)
