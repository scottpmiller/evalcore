"""Built-in classification (aggregate) grader.

For structured-verdict suites - e.g. a moderation or intent classifier -
where each case has a ground-truth label. Computes the set-level metrics that
matter operationally: precision/recall/F1, and the two error rates a security
gate lives or dies on - false-negative rate (a malicious item passed) and
false-positive rate (a benign item blocked).

The ``positive`` class is the one we care about catching (e.g. malicious +
suspicious). Predicted and expected labels are mapped to positive/negative via
the configured label sets, so the grader is reusable for any binary verdict.
"""

from evalcore import models, refs
from evalcore.graders import base


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@base.register('classification', base.GraderType.STATISTICAL)
class Classification:
    """Binary precision/recall/F1 + FN/FP rates over a labeled dataset."""

    def __init__(
        self,
        predicted_ref: str,
        expected_ref: str,
        positive_labels: list[str],
        negative_labels: list[str] | None = None,
        name: str = 'classification',
    ):
        self.name = name
        self.predicted_ref = predicted_ref
        self.expected_ref = expected_ref
        self.positive = {label.lower() for label in positive_labels}
        self.negative = {label.lower() for label in (negative_labels or [])}

    def _is_positive(self, label) -> bool | None:
        """Map a raw label to positive (True) / negative (False) / unknown."""
        if label is None:
            return None
        token = str(label).lower()
        if token in self.positive:
            return True
        if token in self.negative:
            return False
        # Unlisted labels default to negative so a stray verdict can't
        # masquerade as a catch; counted via the ``errors`` metric below.
        return False

    def aggregate(
        self, results: list[models.CaseResult]
    ) -> list[models.Score]:
        tp = fp = fn = tn = errors = 0
        for result in results:
            if result.output.error:
                errors += 1
                continue
            context = {
                'output': result.output.fields,
                'case': result.case.model_dump(),
                'expected': result.case.expected or {},
                'input': result.case.input,
            }
            predicted = self._is_positive(
                refs.resolve_ref(context, self.predicted_ref)
            )
            actual = self._is_positive(
                refs.resolve_ref(context, self.expected_ref)
            )
            if predicted is None or actual is None:
                errors += 1
                continue
            if actual and predicted:
                tp += 1
            elif actual and not predicted:
                fn += 1
            elif not actual and predicted:
                fp += 1
            else:
                tn += 1

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        fnr = _safe_div(fn, fn + tp)
        fpr = _safe_div(fp, fp + tn)
        accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

        def agg(metric: str, value: float) -> models.Score:
            return models.Score(
                grader=self.name, metric=metric, value=value, kind='aggregate'
            )

        return [
            agg('precision', precision),
            agg('recall', recall),
            agg('f1', f1),
            agg('false_negative_rate', fnr),
            agg('false_positive_rate', fpr),
            agg('accuracy', accuracy),
            agg('support_positive', float(tp + fn)),
            agg('support_negative', float(tn + fp)),
            agg('errors', float(errors)),
        ]
