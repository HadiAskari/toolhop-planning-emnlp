"""
Dataset Splitting and Evaluation Framework - FIXED VERSION

This script provides:
1. Train/test/val splits with stratification
2. Judge model evaluation metrics
3. Planner model evaluation metrics
4. Out-of-domain evaluation framework

FIXES:
- Handle both 'data' wrapped and unwrapped JSON formats
- Fixed predictions iteration bug

NEW (non-breaking):
- Robust/relaxed parsing for predictions annotations (extract quality_score even if JSON imperfect)
- Optional debug printing for a few parsed annotations

NEW (Option 2):
- 3-class success metrics with abstention/coverage reporting
- Keep old binary-on-confident-subset metrics for backward compatibility
"""

import json
import re
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for judge model"""

    # Quality score metrics
    quality_score_mae: float
    quality_score_rmse: float
    quality_score_correlation: float

    # Success prediction metrics (legacy: binary on confident subset)
    success_prediction_accuracy: float
    success_prediction_f1: float
    success_prediction_precision: float
    success_prediction_recall: float

    # NEW: 3-class success metrics (success/uncertain/failure)
    success_3class_accuracy: float
    success_3class_macro_f1: float
    success_3class_weighted_f1: float
    success_3class_precision_by_class: Dict[str, float]
    success_3class_recall_by_class: Dict[str, float]
    success_3class_f1_by_class: Dict[str, float]

    # NEW: abstention/coverage stats
    success_coverage: float  # fraction of predictions that are NOT uncertain
    success_accuracy_on_coverage: float  # accuracy on confident subset (binary mapping)

    # By error type breakdown
    mae_by_error_type: Dict[str, float]
    accuracy_by_error_type: Dict[str, float]

    # Calibration metrics
    expected_calibration_error: float

    # Issue detection metrics (placeholder)
    issue_detection_precision: float
    issue_detection_recall: float
    issue_detection_f1: float

    # Summary statistics
    n_samples: int
    n_correct_predictions: int         # legacy: on confident subset (binary)
    n_total_predictions: int           # legacy: size of confident subset (binary)

    # NEW: counts of each predicted success label
    success_pred_label_counts: Dict[str, int]
    success_true_label_counts: Dict[str, int]

    def to_dict(self):
        return asdict(self)


class DatasetSplitter:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def split_dataset(
        self,
        dataset_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        output_dir: str = "./splits"
    ) -> Tuple[str, str, str]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        if isinstance(dataset, dict) and 'data' in dataset:
            data_list = dataset['data']
            metadata = dataset.get('metadata', {})
        elif isinstance(dataset, list):
            data_list = dataset
            metadata = {}
        else:
            raise ValueError("Dataset must be a list or a dict with 'data' key")

        by_query = defaultdict(list)
        for item in data_list:
            by_query[item['query_id']].append(item)

        query_ids = list(by_query.keys())
        n_queries = len(query_ids)
        random.shuffle(query_ids)

        n_train = int(n_queries * train_ratio)
        n_val = int(n_queries * val_ratio)
        train_query_ids = query_ids[:n_train]
        val_query_ids = query_ids[n_train:n_train + n_val]
        test_query_ids = query_ids[n_train + n_val:]

        print(f"Dataset split:")
        print(f"  Total queries: {n_queries}")
        print(f"  Train: {len(train_query_ids)} queries ({len(train_query_ids)/n_queries*100:.1f}%)")
        print(f"  Val:   {len(val_query_ids)} queries ({len(val_query_ids)/n_queries*100:.1f}%)")
        print(f"  Test:  {len(test_query_ids)} queries ({len(test_query_ids)/n_queries*100:.1f}%)")

        train_data = [item for qid in train_query_ids for item in by_query[qid]]
        val_data = [item for qid in val_query_ids for item in by_query[qid]]
        test_data = [item for qid in test_query_ids for item in by_query[qid]]

        print(f"\nPlan counts:")
        print(f"  Train: {len(train_data)} plans")
        print(f"  Val:   {len(val_data)} plans")
        print(f"  Test:  {len(test_data)} plans")

        self._print_error_distribution(train_data, "Train")
        self._print_error_distribution(val_data, "Val")
        self._print_error_distribution(test_data, "Test")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        train_path = f"{output_dir}/train_split.json"
        val_path = f"{output_dir}/val_split.json"
        test_path = f"{output_dir}/test_split.json"

        self._save_split(train_data, train_path, metadata, "train")
        self._save_split(val_data, val_path, metadata, "val")
        self._save_split(test_data, test_path, metadata, "test")

        split_info = {
            'random_seed': self.random_seed,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'train_queries': train_query_ids,
            'val_queries': val_query_ids,
            'test_queries': test_query_ids,
            'train_plans': len(train_data),
            'val_plans': len(val_data),
            'test_plans': len(test_data)
        }

        with open(f"{output_dir}/split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"\n✓ Splits saved to {output_dir}/")
        return train_path, val_path, test_path

    def _print_error_distribution(self, data: List[Dict], split_name: str):
        error_counts = defaultdict(int)
        for item in data:
            error_type = item.get('plan', {}).get('error_type', 'none')
            error_counts[error_type] += 1

        print(f"\n{split_name} error distribution:")
        total = len(data)
        for error_type, count in sorted(error_counts.items()):
            pct = 100 * count / total
            print(f"  {error_type:25s}: {count:5d} ({pct:5.1f}%)")

    def _save_split(self, data: List[Dict], path: str, metadata: Dict, split_name: str):
        split_dataset = {
            'metadata': {
                **metadata,
                'split': split_name,
                'n_plans': len(data)
            },
            'data': data
        }

        with open(path, 'w') as f:
            json.dump(split_dataset, f, indent=2)


class JudgeEvaluator:
    def __init__(self, ground_truth_type: str = "annotations"):
        self.ground_truth_type = ground_truth_type

    # -------------------------
    # Robust annotation parsing
    # -------------------------
    def _normalize_annotation(self, ann: Any) -> Dict[str, Any]:
        if isinstance(ann, dict):
            out = dict(ann)
        else:
            out = {}

        if 'quality_score' not in out:
            out['quality_score'] = 50
        if 'success_prediction' not in out:
            out['success_prediction'] = 'uncertain'
        if 'reasoning' not in out:
            out['reasoning'] = 'No reasoning provided'
        if 'issues' not in out or not isinstance(out.get('issues'), list):
            out['issues'] = []
        if 'confidence' not in out:
            out['confidence'] = 0.5

        try:
            out['quality_score'] = int(out['quality_score'])
        except Exception:
            out['quality_score'] = 50
        out['quality_score'] = max(0, min(100, out['quality_score']))

        try:
            out['confidence'] = float(out['confidence'])
        except Exception:
            out['confidence'] = 0.5
        out['confidence'] = max(0.0, min(1.0, out['confidence']))

        valid_predictions = {'yes', 'likely_yes', 'uncertain', 'likely_no', 'no'}
        if out['success_prediction'] not in valid_predictions:
            out['success_prediction'] = 'uncertain'

        if not isinstance(out.get('reasoning'), str):
            out['reasoning'] = str(out.get('reasoning', ''))

        return out

    def _extract_quality_score_from_any(self, obj: Any) -> Optional[int]:
        text = None
        if isinstance(obj, str):
            text = obj
        elif isinstance(obj, dict):
            for k in ['raw', 'raw_text', 'generated', 'text', 'output']:
                if k in obj and isinstance(obj[k], str):
                    text = obj[k]
                    break

        if not text:
            return None

        m = re.search(r'"quality_score"\s*:\s*([0-9]{1,3})', text)
        if not m:
            m = re.search(r'\bquality_score\b\s*:\s*([0-9]{1,3})', text)
        if m:
            try:
                return max(0, min(100, int(m.group(1))))
            except Exception:
                return None
        return None

    def _relax_annotation(self, ann: Any) -> Dict[str, Any]:
        out = self._normalize_annotation(ann)
        if out.get('quality_score', 50) == 50:
            recovered = self._extract_quality_score_from_any(ann)
            if recovered is not None:
                out['quality_score'] = recovered
                out['confidence'] = min(1.0, max(out.get('confidence', 0.5), 0.2))
                out['reasoning'] = (out.get('reasoning', '') or '') + " (fallback: extracted quality_score)"
        return out

    # -------------------------
    # Success mapping helpers
    # -------------------------
    @staticmethod
    def _success_to_binary(success_pred: str) -> int:
        """1=success, 0=failure, -1=uncertain"""
        if success_pred in ['yes', 'likely_yes']:
            return 1
        elif success_pred in ['no', 'likely_no']:
            return 0
        else:
            return -1

    @staticmethod
    def _success_to_3class(success_pred: str) -> str:
        """Map to {success, uncertain, failure}"""
        if success_pred in ['yes', 'likely_yes']:
            return 'success'
        elif success_pred in ['no', 'likely_no']:
            return 'failure'
        else:
            return 'uncertain'

    @staticmethod
    def _compute_prf_per_class(y_true: List[str], y_pred: List[str], labels: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Compute per-class precision/recall/f1 (one-vs-rest) with no sklearn.
        """
        prec, rec, f1 = {}, {}, {}
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for lab in labels:
            tp = np.sum((y_pred == lab) & (y_true == lab))
            fp = np.sum((y_pred == lab) & (y_true != lab))
            fn = np.sum((y_pred != lab) & (y_true == lab))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            prec[lab] = float(p)
            rec[lab] = float(r)
            f1[lab] = float(f)

        return prec, rec, f1

    def evaluate(
        self,
        predictions_path: str,
        ground_truth_path: str,
        output_path: Optional[str] = None
    ) -> EvaluationMetrics:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)

        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        if isinstance(predictions, dict) and 'data' in predictions:
            pred_list = predictions['data']
        elif isinstance(predictions, list):
            pred_list = predictions
        else:
            raise ValueError("Predictions must be a list or dict with 'data' key")

        if isinstance(ground_truth, dict) and 'data' in ground_truth:
            gt_list = ground_truth['data']
        elif isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            raise ValueError("Ground truth must be a list or dict with 'data' key")

        pred_by_id = {}
        pred_parse_stats = {
            "n_predictions": 0,
            "n_missing_annotation": 0,
            "n_relaxed_fallback_used": 0,
        }

        for item in pred_list:
            pred_parse_stats["n_predictions"] += 1

            steps_tuple = tuple(
                (s['step_id'], s['tool_name'])
                for s in sorted(item['plan']['steps'], key=lambda x: x['step_id'])
            )
            key = (item['query_id'], steps_tuple)

            if 'annotation' not in item:
                pred_parse_stats["n_missing_annotation"] += 1
                pred_ann = self._normalize_annotation({})
            else:
                pred_ann_raw = item['annotation']
                pred_ann = self._relax_annotation(pred_ann_raw)

                if isinstance(pred_ann_raw, dict):
                    if 'quality_score' not in pred_ann_raw:
                        pred_parse_stats["n_relaxed_fallback_used"] += 1
                elif isinstance(pred_ann_raw, str):
                    pred_parse_stats["n_relaxed_fallback_used"] += 1

            pred_by_id[key] = pred_ann

        quality_scores_pred = []
        quality_scores_true = []
        success_preds_bin = []
        success_true_bin = []
        success_preds_3 = []
        success_true_3 = []
        error_types = []
        confidence_scores = []

        by_error_type = defaultdict(lambda: {'pred': [], 'true': []})

        matched = 0
        for item in gt_list:
            steps_tuple = tuple(
                (s['step_id'], s['tool_name'])
                for s in sorted(item['plan']['steps'], key=lambda x: x['step_id'])
            )
            key = (item['query_id'], steps_tuple)

            if key not in pred_by_id:
                print(f"Warning: No prediction found for query_id {item['query_id']}")
                continue

            matched += 1
            pred_ann = pred_by_id[key]
            true_ann = self._normalize_annotation(item.get('annotation', {}))

            quality_scores_pred.append(pred_ann['quality_score'])
            quality_scores_true.append(true_ann['quality_score'])

            # legacy binary mapping
            success_preds_bin.append(self._success_to_binary(pred_ann['success_prediction']))
            success_true_bin.append(self._success_to_binary(true_ann['success_prediction']))

            # new 3-class mapping
            success_preds_3.append(self._success_to_3class(pred_ann['success_prediction']))
            success_true_3.append(self._success_to_3class(true_ann['success_prediction']))

            error_type = item['plan'].get('error_type', 'none')
            error_types.append(error_type)
            by_error_type[error_type]['pred'].append(pred_ann['quality_score'])
            by_error_type[error_type]['true'].append(true_ann['quality_score'])

            confidence_scores.append(pred_ann.get('confidence', 0.5))

        print(f"\nMatched {matched}/{len(gt_list)} ground truth items with predictions")
        print("Prediction annotation parse stats:")
        print(f"  n_predictions_loaded:        {pred_parse_stats['n_predictions']}")
        print(f"  n_missing_annotation:        {pred_parse_stats['n_missing_annotation']}")
        print(f"  n_relaxed_fallback_used:     {pred_parse_stats['n_relaxed_fallback_used']}")

        # Print success label distributions (super useful for debugging)
        pred_counts = dict(defaultdict(int))
        true_counts = dict(defaultdict(int))
        for x in success_preds_3:
            pred_counts[x] = pred_counts.get(x, 0) + 1
        for x in success_true_3:
            true_counts[x] = true_counts.get(x, 0) + 1

        print("\nSuccess label distribution (3-class):")
        print(f"  Pred: {pred_counts}")
        print(f"  True: {true_counts}")

        if matched == 0:
            raise ValueError("No predictions matched ground truth! Check data format.")

        metrics = self._compute_metrics(
            quality_scores_pred, quality_scores_true,
            success_preds_bin, success_true_bin,
            success_preds_3, success_true_3,
            by_error_type,
            confidence_scores,
            pred_counts,
            true_counts
        )

        self._print_metrics(metrics)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"\n✓ Detailed results saved to {output_path}")

        return metrics

    def _compute_metrics(
        self,
        quality_pred: List[float],
        quality_true: List[float],
        success_pred_bin: List[int],
        success_true_bin: List[int],
        success_pred_3: List[str],
        success_true_3: List[str],
        by_error_type: Dict,
        confidence: List[float],
        pred_counts: Dict[str, int],
        true_counts: Dict[str, int],
    ) -> EvaluationMetrics:
        quality_pred = np.array(quality_pred)
        quality_true = np.array(quality_true)

        mae = np.mean(np.abs(quality_pred - quality_true))
        rmse = np.sqrt(np.mean((quality_pred - quality_true) ** 2))
        correlation = np.corrcoef(quality_pred, quality_true)[0, 1] if len(quality_pred) > 1 else 0.0

        # ----------------------------
        # Legacy binary metrics (confident subset only)
        # ----------------------------
        success_pred_binary = [p for p in success_pred_bin if p != -1]
        success_true_binary = [t for i, t in enumerate(success_true_bin) if success_pred_bin[i] != -1]

        if len(success_pred_binary) > 0:
            accuracy_bin = float(np.mean(np.array(success_pred_binary) == np.array(success_true_binary)))

            tp = sum((p == 1 and t == 1) for p, t in zip(success_pred_binary, success_true_binary))
            fp = sum((p == 1 and t == 0) for p, t in zip(success_pred_binary, success_true_binary))
            fn = sum((p == 0 and t == 1) for p, t in zip(success_pred_binary, success_true_binary))

            precision_bin = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_bin = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_bin = (2 * precision_bin * recall_bin / (precision_bin + recall_bin)) if (precision_bin + recall_bin) > 0 else 0.0

            n_correct_bin = int(np.sum(np.array(success_pred_binary) == np.array(success_true_binary)))
            n_total_bin = len(success_pred_binary)
        else:
            accuracy_bin = precision_bin = recall_bin = f1_bin = 0.0
            n_correct_bin = 0
            n_total_bin = 0

        # ----------------------------
        # NEW: 3-class metrics
        # ----------------------------
        labels_3 = ['success', 'uncertain', 'failure']
        y_true = list(success_true_3)
        y_pred = list(success_pred_3)

        acc_3 = float(np.mean(np.array(y_true) == np.array(y_pred)))

        prec_by, rec_by, f1_by = self._compute_prf_per_class(y_true, y_pred, labels_3)
        macro_f1 = float(np.mean([f1_by[l] for l in labels_3]))

        # weighted f1
        supports = {l: sum(1 for t in y_true if t == l) for l in labels_3}
        total = len(y_true)
        weighted_f1 = float(sum((supports[l] / total) * f1_by[l] for l in labels_3)) if total > 0 else 0.0

        # coverage = non-uncertain fraction in predictions
        n_non_uncertain = sum(1 for p in y_pred if p != 'uncertain')
        coverage = float(n_non_uncertain / len(y_pred)) if len(y_pred) > 0 else 0.0

        # accuracy on coverage (binary mapping among confident subset)
        # Use the same confident subset used by legacy metrics
        acc_on_cov = float(accuracy_bin)

        # By error type metrics
        mae_by_error_type = {}
        accuracy_by_error_type = {}
        for error_type, scores in by_error_type.items():
            if len(scores['pred']) > 0:
                pred_arr = np.array(scores['pred'])
                true_arr = np.array(scores['true'])
                mae_by_error_type[error_type] = float(np.mean(np.abs(pred_arr - true_arr)))
                accuracy_by_error_type[error_type] = float(np.mean(np.abs(pred_arr - true_arr) <= 15))

        ece = self._compute_ece(quality_pred, quality_true, confidence)

        issue_precision = issue_recall = issue_f1 = 0.0

        return EvaluationMetrics(
            quality_score_mae=float(mae),
            quality_score_rmse=float(rmse),
            quality_score_correlation=float(correlation),

            # legacy binary (confident subset)
            success_prediction_accuracy=float(accuracy_bin),
            success_prediction_f1=float(f1_bin),
            success_prediction_precision=float(precision_bin),
            success_prediction_recall=float(recall_bin),

            # 3-class
            success_3class_accuracy=float(acc_3),
            success_3class_macro_f1=float(macro_f1),
            success_3class_weighted_f1=float(weighted_f1),
            success_3class_precision_by_class=prec_by,
            success_3class_recall_by_class=rec_by,
            success_3class_f1_by_class=f1_by,

            # coverage
            success_coverage=float(coverage),
            success_accuracy_on_coverage=float(acc_on_cov),

            mae_by_error_type=mae_by_error_type,
            accuracy_by_error_type=accuracy_by_error_type,
            expected_calibration_error=float(ece),
            issue_detection_precision=issue_precision,
            issue_detection_recall=issue_recall,
            issue_detection_f1=issue_f1,

            n_samples=len(quality_pred),
            n_correct_predictions=n_correct_bin,
            n_total_predictions=n_total_bin,

            success_pred_label_counts={k: int(v) for k, v in pred_counts.items()},
            success_true_label_counts={k: int(v) for k, v in true_counts.items()},
        )

    def _compute_ece(self, pred: np.ndarray, true: np.ndarray, conf: List[float]) -> float:
        if len(conf) == 0:
            return 0.0
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        conf_arr = np.array(conf)
        for i in range(n_bins):
            mask = (conf_arr >= bins[i]) & (conf_arr < bins[i + 1])
            if np.sum(mask) > 0:
                avg_conf = float(np.mean(conf_arr[mask]))
                avg_acc = float(np.mean(np.abs(pred[mask] - true[mask]) <= 15))
                ece += float(np.sum(mask) / len(conf) * np.abs(avg_conf - avg_acc))
        return float(ece)

    def _print_metrics(self, metrics: EvaluationMetrics):
        print("\n" + "=" * 80)
        print("JUDGE MODEL EVALUATION RESULTS")
        print("=" * 80)

        print("\n📊 Quality Score Metrics:")
        print(f"  Mean Absolute Error (MAE):  {metrics.quality_score_mae:.2f}")
        print(f"  Root Mean Squared Error:    {metrics.quality_score_rmse:.2f}")
        print(f"  Pearson Correlation:        {metrics.quality_score_correlation:.3f}")

        print("\n🎯 Success Prediction (Legacy binary, confident subset only):")
        print(f"  Coverage (non-uncertain): {metrics.success_coverage:.3f}")
        print(f"  Accuracy on coverage:     {metrics.success_accuracy_on_coverage:.3f}")
        print(f"  Precision:               {metrics.success_prediction_precision:.3f}")
        print(f"  Recall:                  {metrics.success_prediction_recall:.3f}")
        print(f"  F1 Score:                {metrics.success_prediction_f1:.3f}")
        print(f"  Correct (coverage only): {metrics.n_correct_predictions}/{metrics.n_total_predictions}")

        print("\n🧭 Success Prediction (NEW 3-class: success/uncertain/failure):")
        print(f"  3-class Accuracy:        {metrics.success_3class_accuracy:.3f}")
        print(f"  Macro-F1:                {metrics.success_3class_macro_f1:.3f}")
        print(f"  Weighted-F1:             {metrics.success_3class_weighted_f1:.3f}")
        print(f"  Per-class Precision:     {metrics.success_3class_precision_by_class}")
        print(f"  Per-class Recall:        {metrics.success_3class_recall_by_class}")
        print(f"  Per-class F1:            {metrics.success_3class_f1_by_class}")
        print(f"  Pred label counts:       {metrics.success_pred_label_counts}")
        print(f"  True label counts:       {metrics.success_true_label_counts}")

        print("\n📈 MAE by Error Type:")
        for error_type, mae in sorted(metrics.mae_by_error_type.items()):
            accuracy = metrics.accuracy_by_error_type.get(error_type, 0)
            print(f"  {error_type:25s}: MAE={mae:5.2f}, Acc@15={accuracy:.3f}")

        print(f"\n⚖️  Calibration:")
        print(f"  Expected Calibration Error: {metrics.expected_calibration_error:.3f}")

        print(f"\n📝 Dataset Info:")
        print(f"  Total samples evaluated: {metrics.n_samples}")

        print("\n" + "=" * 80)

    # end class


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset splitting and evaluation")
    parser.add_argument('--action', choices=['split', 'evaluate'], required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./evaluation')
    parser.add_argument('--predictions-path', type=str, help='For evaluation')

    args = parser.parse_args()

    if args.action == 'split':
        splitter = DatasetSplitter(random_seed=42)
        train_path, val_path, test_path = splitter.split_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        print(f"\n✓ Splits created:")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")
        print(f"  Test:  {test_path}")

    elif args.action == 'evaluate':
        if not args.predictions_path:
            print("Error: --predictions-path required for evaluation")
            return

        evaluator = JudgeEvaluator()
        _ = evaluator.evaluate(
            predictions_path=args.predictions_path,
            ground_truth_path=args.dataset_path,
            output_path=f"{args.output_dir}/evaluation_results.json"
        )


if __name__ == '__main__':
    main()