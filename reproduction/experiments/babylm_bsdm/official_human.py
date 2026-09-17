# Adapted from the pinned BabyLM evaluator: local paths and structured results.
# See reproduction/THIRD_PARTY_NOTICES.md and reproduction/BABYLM_LICENSE.
# Exact functions from babylm-org/babylm-eval 6f825c291e2c4c78ad33b1935fd64d45f52642dc
from __future__ import annotations
from pathlib import Path
from typing import Any
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
logger=logging.getLogger(__name__)

def _calculate_reading_results(results_dict: dict[str, dict[str, list[dict[str, int | float]]]], path_to_data: Path) -> dict[str, float]:
    data = pd.read_csv(path_to_data, dtype={'item': str})
    preds = [item["pred"] for item in results_dict["reading"]["predictions"]]
    prev_preds = [item["prev_pred"] for item in results_dict["reading"]["predictions"]]
    data["pred"] = preds
    data["prev_pred"] = prev_preds
    eye_tracking_vars = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound']
    eye_tracking_result = []
    for dv in eye_tracking_vars:
        # Baseline model
        temp = data[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        # Predictive model
        temp = data[[dv, "Subtlex_log10", "length", "context_length", "pred"]].dropna()
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        R2_model = float(OLS_model.rsquared)
        eye_tracking_result.append(((R2_model-R2_baseline)/(1-R2_baseline)))
    eye_tracking_result = sum(eye_tracking_result) / len(eye_tracking_result)

    # Baseline model
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
    OLS_baseline = smf.ols(formula='self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred', data=temp).fit()
    R2_baseline = float(OLS_baseline.rsquared)
    # Predictive model
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred", "pred"]].dropna()
    OLS_model = smf.ols(formula='self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred', data=temp).fit()
    R2_model = float(OLS_model.rsquared)
    self_paced_reading_result = ((R2_model-R2_baseline)/(1-R2_baseline))

    processed_results = {"spr": self_paced_reading_result, "rt": eye_tracking_result}

    return processed_results

def sigmoid_function(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """Sigmoid function for fitting learning curves: f(x) = a / (1 + exp(-b*(x-c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d

class AoAEvaluator:
    """Evaluates Age of Acquisition based on Chang & Bergen 2022 methodology."""

    def __init__(self, cdi_data_path: Path):
        """Initialize with CDI reference data."""
        self.cdi_data = pd.read_csv(cdi_data_path)
        self.prepare_cdi_data()

    def prepare_cdi_data(self) -> None:
        """Prepare CDI data for evaluation."""
        # Extract age columns (16-30 months)
        age_columns = [str(age) for age in range(16, 31)]
        available_age_columns = [
            col for col in age_columns if col in self.cdi_data.columns
        ]

        if not available_age_columns:
            raise ValueError("No age columns found in CDI data")

        # Create age mapping
        self.ages = np.array([int(col) for col in available_age_columns])
        self.age_data = self.cdi_data[available_age_columns].values
        self.words = self.cdi_data["word"].values

        logger.info(
            f"Loaded CDI data for {len(self.words)} words across ages {self.ages[0]}-{self.ages[-1]} months"
        )

    def compute_child_aoa(self, word_idx: int, threshold: float = 0.5) -> float | None:
        """Compute Age of Acquisition for a word in children (50% threshold)."""
        proportions = self.age_data[word_idx]

        # Remove NaN values
        valid_mask = ~np.isnan(proportions)
        if not np.any(valid_mask):
            return None

        valid_ages = self.ages[valid_mask]
        valid_proportions = proportions[valid_mask]

        # Check if we have enough data points
        if len(valid_proportions) < 3:
            return None

        try:
            # Fit sigmoid curve
            # Initial parameter guesses: [a, b, c, d]
            initial_guess = [1.0, 0.1, np.mean(valid_ages), 0.0]
            bounds = ([0, 0, valid_ages[0], -0.5], [2.0, 1.0, valid_ages[-1], 0.5])

            popt, _ = curve_fit(
                sigmoid_function,
                valid_ages,
                valid_proportions,
                p0=initial_guess,
                bounds=bounds,
                maxfev=1000,
            )

            # Find age where sigmoid reaches threshold
            # Solve: threshold = a / (1 + exp(-b*(x-c))) + d
            a, b, c, d = popt
            if b <= 0 or a <= 0:
                return None

            # Rearrange sigmoid equation to solve for x
            y_target = threshold
            if y_target <= d or y_target >= a + d:
                return None

            aoa = c - np.log((a / (y_target - d)) - 1) / b

            # Ensure AoA is within reasonable bounds
            if aoa < valid_ages[0] or aoa > valid_ages[-1]:
                return None

            return aoa

        except Exception as e:
            logger.debug(f"Failed to fit sigmoid for word {self.words[word_idx]}: {e}")
            return None

    def compute_model_aoa(
        self,
        surprisal_data: list[float],
        training_steps: list[int],
        vocab_size: int,
        n_subword_tokens: int = 1,
        threshold_percentile: float = 0.5,
    ) -> float | None:
        """
        Compute Age of Acquisition for a word in the model.
        Based on Chang & Bergen 2022: 50% between random chance and minimum surprisal.
        """
        if len(surprisal_data) != len(training_steps):
            raise ValueError("Surprisal data and training steps must have same length")

        if len(surprisal_data) < 3:
            return None

        # Convert to numpy arrays
        steps = np.array(training_steps)
        surprisals = np.array(surprisal_data)

        # Remove NaN values
        valid_mask = ~np.isnan(surprisals)
        if not np.any(valid_mask):
            return None

        valid_steps = steps[valid_mask]
        valid_surprisals = surprisals[valid_mask]

        # Uniform-random baseline in nats. Surprisal is summed over the word's subword
        # tokens, so the ceiling is summed too: n_subword_tokens * ln(vocab_size).
        random_chance_surprisal = n_subword_tokens * np.log(vocab_size)
        min_surprisal = np.min(valid_surprisals)

        # Calculate threshold surprisal (50% between random chance and minimum)
        threshold_surprisal = random_chance_surprisal - threshold_percentile * (
            random_chance_surprisal - min_surprisal
        )

        try:
            # Fit sigmoid to learning curve (note: surprisal decreases over time)
            # We'll fit to negative surprisal to get increasing curve
            neg_surprisals = -valid_surprisals

            # Use log scale for training steps
            log_steps = np.log10(valid_steps + 1)  # +1 to handle step 0

            # Bounded fit: the old unbounded maxfev=1000 fit let b blow up and failed to
            # converge on many clean decreasing curves. Bounds + more evals recover them.
            rng = np.max(neg_surprisals) - np.min(neg_surprisals)
            initial_guess = [
                rng,  # a: range
                1.0,  # b: steepness
                np.mean(log_steps),  # c: midpoint
                np.min(neg_surprisals),  # d: offset
            ]
            lower = [0.0, 0.0, np.min(log_steps) - 1, np.min(neg_surprisals) - 2 * rng - 1]
            upper = [10 * rng + 1, 100.0, np.max(log_steps) + 1, np.max(neg_surprisals) + 1]

            popt, _ = curve_fit(
                sigmoid_function,
                log_steps,
                neg_surprisals,
                p0=initial_guess,
                bounds=(lower, upper),
                maxfev=20000,
            )

            # Find step where surprisal reaches threshold
            a, b, c, d = popt
            neg_threshold = -threshold_surprisal

            if b <= 1e-6 or a <= 1e-6:
                return None

            if neg_threshold <= d or neg_threshold >= a + d:
                return None

            log_aoa_step = c - np.log((a / (neg_threshold - d)) - 1) / b
            aoa_step = 10**log_aoa_step - 1

            # Ensure AoA is within reasonable bounds
            if aoa_step < valid_steps[0] or aoa_step > valid_steps[-1]:
                return None

            return log_aoa_step

        except Exception as e:
            logger.debug(f"Failed to fit sigmoid for model surprisal: {e}")
            return None

    def extract_step_number(self, step_name: str) -> float | None:
        """Extract numeric step value from step name (e.g., 'chck_10M' -> 10000000)."""
        import re

        # Handle different step name formats
        if isinstance(step_name, (int, float)):
            return float(step_name)

        # Extract number and unit from step name
        match = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", str(step_name), re.IGNORECASE)
        if not match:
            return None

        number = float(match.group(1))
        unit = match.group(2).upper() if match.group(2) else ""

        # Convert to actual step count
        multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}
        multiplier = multipliers.get(unit, 1)

        return number * multiplier

    def compute_curve_fitness(
        self, model_results: dict[str, t.Any], tokenizer: t.Any,
        target_words: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute curve fitness scores comparing model and child acquisition.
        Returns correlation between model and child AoAs.
        """
        results = model_results.get("results", [])
        if not results:
            raise ValueError("No results found in model data")

        vocab_size = tokenizer.vocab_size
        # Subword length of a word in context: tokenize after a fixed prefix and subtract
        # it, cancelling the tokenizer's leading-space handling.
        prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]

        def subword_len(word: str) -> int:
            ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
            return max(1, len(ids) - len(prefix_ids))

        # Organize data by word
        word_data = {}
        for result in results:
            word = result["target_word"]
            if target_words and word not in target_words:
                continue

            if word not in word_data:
                word_data[word] = {"steps": [], "surprisals": []}

            # Extract numeric step value from step name
            try:
                step_val = self.extract_step_number(result["step"])
                surprisal_val = float(result["surprisal"])

                if step_val is not None:
                    word_data[word]["steps"].append(step_val)
                    word_data[word]["surprisals"].append(surprisal_val)
                else:
                    logger.warning(
                        f"Could not extract step number from: {result['step']}"
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid data for word {word}: {e}")
                continue

        # Compute AoAs for both model and children
        model_aoas = []
        child_aoas = []
        valid_words = []

        for word in word_data.keys():
            # Find word in CDI data
            word_mask = self.cdi_data["word"] == word
            if not np.any(word_mask):
                continue

            word_idx = np.where(word_mask)[0][0]

            # Compute child AoA
            child_aoa = self.compute_child_aoa(word_idx)
            if child_aoa is None:
                continue

            # Average the per-context surprisals within each checkpoint, so the fit and
            # the threshold floor use one denoised point per step rather than the noisy
            # per-context cloud (a single low context otherwise strands the threshold).
            steps_arr = np.array(word_data[word]["steps"])
            surp_arr = np.array(word_data[word]["surprisals"])
            uniq_steps = np.unique(steps_arr)
            mean_surprisals = [float(surp_arr[steps_arr == s].mean()) for s in uniq_steps]

            # Compute model AoA (ceiling scaled by the word's subword length)
            model_aoa = self.compute_model_aoa(
                mean_surprisals, uniq_steps.tolist(),
                vocab_size, n_subword_tokens=subword_len(word),
            )
            if model_aoa is None:
                continue

            model_aoas.append(model_aoa)
            child_aoas.append(child_aoa)
            valid_words.append(word)

        if len(model_aoas) < 3:
            logger.warning(f"Only {len(model_aoas)} valid words for correlation")
            return {"curve_fitness": 0.0, "n_words": len(model_aoas)}

        # Compute correlation (curve fitness)
        correlation, p_value = pearsonr(model_aoas, child_aoas)

        if p_value > 0.1:
            return {"curve_fitness": 0.0, "n_words": len(model_aoas)}

        # Compute mean monthly scores (simplified metric)
        mean_monthly_scores = {}
        for age in range(16, 31):
            if str(age) in self.cdi_data.columns:
                age_scores = self.cdi_data[str(age)].dropna()
                mean_monthly_scores[f"month_{age}"] = float(age_scores.mean())

        overall_mean_score = np.mean(list(mean_monthly_scores.values()))

        logger.info(f"Model-Child AoA correlation: {correlation:.3f} (p={p_value:.3f})")
        logger.info(f"Overall mean monthly score: {overall_mean_score:.3f}")

        return {
            "curve_fitness": float(correlation),
            "p_value": float(p_value),
            "n_words": len(valid_words),
            "mean_monthly_score": overall_mean_score,
            "monthly_scores": mean_monthly_scores,
            "valid_words": valid_words,
            "model_aoas": model_aoas,
            "child_aoas": child_aoas,
        }
