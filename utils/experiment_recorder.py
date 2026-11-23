import json
from datetime import datetime, timezone
from typing import Dict, List, Optional


class ExperimentRecorder:
    def __init__(self, innovation_names: List[str], sota_baselines: Dict[str, float],
                 baseline_metrics: Dict[str, float], iteration_round: int,
                 dataset_name: str = ""):
        self.innovation_names = innovation_names
        self.sota_baselines = sota_baselines
        self.baseline_metrics = baseline_metrics
        self.iteration_round = iteration_round
        self.dataset_name = dataset_name
        self.metrics: Dict[str, float] = {}
        self.ablation_results: List[Dict] = []
        self.failures: List[Dict] = []

    def record_metric(self, name: str, value: float):
        self.metrics[name] = value

    def record_ablation_batch(self, innovation_name: str,
                              with_scores: List[float], without_scores: List[float],
                              metric_name: str = "dice",
                              claimed_improvement: str = ""):
        n = len(with_scores)
        import numpy as np
        mean_with = float(np.mean(with_scores))
        mean_without = float(np.mean(without_scores))
        std_with = float(np.std(with_scores)) if n > 1 else 0.0
        std_without = float(np.std(without_scores)) if n > 1 else 0.0
        cohens_d = None
        interpretation = "insufficient_data"
        if n >= 3:
            pooled_std = np.sqrt((std_with ** 2 + std_without ** 2) / 2.0)
            if pooled_std > 1e-8:
                cohens_d = float((mean_with - mean_without) / pooled_std)
            else:
                cohens_d = 0.0
            interpretation = self._interpret_effect_size(cohens_d)
        effectiveness = self._compute_effectiveness(cohens_d) if cohens_d is not None else "insufficient_data"
        self.ablation_results.append({
            "innovation": innovation_name,
            "metric_name": metric_name,
            "mean_with": mean_with,
            "mean_without": mean_without,
            "std_with": std_with,
            "std_without": std_without,
            "n": n,
            "effect_size": {"cohens_d": cohens_d, "interpretation": interpretation},
            "effectiveness": effectiveness,
            "claimed_improvement": claimed_improvement,
        })

    def _compute_effectiveness(self, cohens_d: float) -> str:
        if cohens_d >= 0.8:
            return "valid"
        elif cohens_d >= 0.3:
            return "partial"
        else:
            return "invalid"

    @staticmethod
    def _interpret_effect_size(cohens_d: float) -> str:
        if cohens_d >= 0.8:
            return "large"
        elif cohens_d >= 0.5:
            return "medium"
        elif cohens_d >= 0.2:
            return "small"
        elif cohens_d >= -0.2:
            return "negligible"
        else:
            return "negative"

    def record_failure(self, scenario: str, effect: str, analysis: str = ""):
        self.failures.append({
            "scenario": scenario,
            "effect": effect,
            "analysis": analysis,
        })

    def save(self, output_path: str = "experiment_summary.json"):
        overall = {}
        for k, v in self.metrics.items():
            overall[k] = v
        overall["dataset_name"] = self.dataset_name
        overall["sota_baselines"] = dict(self.sota_baselines)
        convergence = self._compute_convergence()
        summary = {
            "iteration_round": self.iteration_round,
            "overall_metrics": overall,
            "innovation_results": self.ablation_results,
            "convergence": convergence,
            "failure_cases": self.failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    def _compute_convergence(self) -> Dict:
        exceeds_sota = True
        gap_to_sota = None
        higher_is_better = {"dice", "iou", "map", "accuracy", "recall", "precision"}
        for metric_key, sota_val in self.sota_baselines.items():
            best_key = f"best_{metric_key}" if not metric_key.startswith("best_") else metric_key
            our_val = self.metrics.get(best_key, self.metrics.get(metric_key))
            if our_val is None:
                continue
            if any(h in metric_key for h in ["dice", "iou", "acc", "map"]):
                if our_val < sota_val - 1e-6:
                    exceeds_sota = False
            else:
                if our_val > sota_val + 1e-6:
                    exceeds_sota = False
            if gap_to_sota is None:
                gap_to_sota = abs(our_val - sota_val)
        return {
            "exceeds_sota": exceeds_sota,
            "gap_to_sota": gap_to_sota,
            "performance_trend": "unknown",
        }
