"""
Hybrid Water Anomaly Detection Model for Apartment Buildings  (v2)

DESCRIPTION:
    Implements a hybrid anomaly detector for building-level water flow data, combining
    statistical CUSUM change detection with machine learning Isolation Forest.
    Designed for 50-apartment building aggregate flow monitoring.

KEY CHANGES IN v2:
    - 7 features (was 5): added flow_trend + baseline_elev for sustained-leak detection
    - flow_trend: linear regression slope of 20-min window → catches rising baselines
    - baseline_elev: normalized deviation of inter-appliance mean from historical norm
    - Rolling 60-min baseline tracker for live inference of baseline_elev
    - Fixed CUSUM: removed zero-flow decay, reduced k (more sensitive to small leaks)
      Appliance resets now only partial (×0.5 decay, not full reset), preserving
      accumulated evidence of ongoing small leaks
    - Updated default thresholds to match v2 calibration
    - decision_threshold lowered to 0.55 (was 0.65) for better small-leak recall

DEPENDENCIES:
    - numpy: Data processing
    - sklearn: Isolation Forest model and scaler (passed in from training script)
"""

import numpy as np
from collections import deque


class HybridWaterAnomalyDetector:
    """Hybrid anomaly detector combining CUSUM and Isolation Forest for building-level detection."""

    def __init__(
        self,
        if_model,
        if_scaler,
        # CUSUM — tuned for sustained small leaks
        cusum_k=0.3,               # Lower k = catches smaller sustained deviations (was 0.5)
        cusum_h=15.0,              # Lower h = triggers sooner (was 20.0)
        noise_floor=0.2,
        # Isolation Forest
        if_threshold=-0.05,
        if_score_scale=0.1,
        # Feature extraction
        appliance_flow_thresh=8.0,
        clip_bound=10.0,
        # Baseline stats for baseline_elev feature (loaded from calibration JSON)
        baseline_inter_mean_median=1.5,
        baseline_inter_mean_std=0.8,
        # Fusion & decision
        w2=0.35,                   # CUSUM weight (was 0.4)
        w3=0.65,                   # IF weight   (was 0.6)
        decision_threshold=0.55,   # Lower threshold → catches sustained leaks (was 0.65)
        persistence_windows=2,
    ):
        """
        Initialize the hybrid anomaly detector.

        Args:
            if_model:                     Trained Isolation Forest model (7 features)
            if_scaler:                    StandardScaler fitted on 7 training features
            cusum_k:                      CUSUM sensitivity parameter
            cusum_h:                      CUSUM trigger threshold
            noise_floor:                  Flow below this treated as zero
            if_threshold:                 IF decision function cut-off (anomaly when score < this)
            if_score_scale:               Scale for normalising IF score to [0,1]
            appliance_flow_thresh:        Flow > this = appliance event (not baseline)
            clip_bound:                   Clips scaled features to [-clip, +clip]
            baseline_inter_mean_median:   Historical median of inter-appliance mean flow
            baseline_inter_mean_std:      Historical std dev of inter-appliance mean flow
            w2, w3:                       Weights for CUSUM and IF in fusion
            decision_threshold:           Fused score threshold for anomaly declaration
            persistence_windows:          Consecutive anomaly windows required
        """
        self.if_model = if_model
        self.if_scaler = if_scaler

        # CUSUM
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.noise_floor = noise_floor

        # IF
        self.if_threshold = if_threshold
        self.if_score_scale = if_score_scale
        self.clip_bound = clip_bound

        # Features
        self.appliance_flow_thresh = appliance_flow_thresh
        self.baseline_inter_mean_median = baseline_inter_mean_median
        self.baseline_inter_mean_std = baseline_inter_mean_std

        # Fusion
        self.w2 = w2
        self.w3 = w3
        self.decision_threshold = decision_threshold
        self.persistence_windows = persistence_windows

        # ── State ──
        self._anomaly_streak = 0
        self.cusum_s = 0.0
        self._prev_appliance = False

        # Rolling 60-minute history for baseline_elev (inter-appliance mean values)
        self._inter_mean_history: deque = deque(maxlen=3)   # 3 × 20-min windows = 60 min

    # ─────────────────────────────────────────────────────────────────────────
    # Level 2: CUSUM  (Fixed for sustained small leaks)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_cusum(self, window):
        """
        Run CUSUM change detection on a window of flow data.

        v2 changes vs v1:
        - Removed zero-flow decay (*0.8). In a 50-apartment building, true zero
          flow is rare; the decay was preventing accumulation of sustained-leak signal.
        - Appliance resets are now PARTIAL (s *= 0.5), not full reset (s = 0).
          This preserves accumulated evidence that a small baseline leak exists
          even when appliances are running on top of it.
        - k lowered from 0.5 to 0.3 → accumulates more signal from small leaks
        """
        s = self.cusum_s
        triggered = False
        prev = self._prev_appliance

        for lpm in window:
            appliance = lpm >= self.appliance_flow_thresh

            # On appliance START: partial decay (not full reset)
            if appliance and not prev:
                s *= 0.5   # Was: s = 0.0  — preserve some leak signal

            # Accumulate for baseline/inter-appliance flow
            if not appliance:
                # Accumulate whenever flow is above noise floor
                # (small sustained leak will keep delta positive)
                delta = lpm - self.cusum_k
                s = max(0.0, s + delta)

                if s >= self.cusum_h:
                    triggered = True

            prev = appliance

        self.cusum_s = s
        self._prev_appliance = prev
        return s, triggered

    # ─────────────────────────────────────────────────────────────────────────
    # Level 3: Feature extraction for Isolation Forest
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_features(self, window):
        """
        Extract 7 statistical features from a window for Isolation Forest.

        Features (must match train_model.py extract_features() exactly):
        1. mnf:           10th percentile of non-zero flows
        2. inter_mean:    Mean of inter-appliance flows
        3. inter_frac:    Fraction of inter-appliance periods above noise floor
        4. mean_flow:     Average window flow
        5. inter_std:     Std dev of inter-appliance flows
        6. flow_trend:    Linear regression slope [L/min per minute]
        7. baseline_elev: (inter_mean - historical_median) / historical_std

        Returns scaled feature vector (1, 7) ready for IF model.
        """
        inter = window[window < self.appliance_flow_thresh]
        nonzero = window[window > 0.0]

        # Feature 1: mnf
        mnf = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.0

        # Feature 2: inter_mean
        inter_mean = float(inter.mean()) if len(inter) > 0 else 0.0

        # Feature 3: inter_frac
        inter_frac = float((inter > self.noise_floor).mean()) if len(inter) > 0 else 0.0

        # Feature 4: mean_flow
        mean_flow = float(window.mean())

        # Feature 5: inter_std
        inter_std = float(inter.std()) if len(inter) > 1 else 0.0

        # Feature 6: flow_trend (linear regression slope)
        t = np.arange(len(window), dtype=np.float64)
        t_mean = t.mean()
        f_mean = window.mean()
        denom = ((t - t_mean) ** 2).sum()
        if denom > 1e-9:
            flow_trend = float(((t - t_mean) * (window.astype(np.float64) - f_mean)).sum() / denom)
        else:
            flow_trend = 0.0

        # Feature 7: baseline_elev — rolling 60-min context
        # Add this window's inter_mean to history, then compute elev vs training stats
        self._inter_mean_history.append(inter_mean)
        # Use rolling average as a smoother estimate of current baseline
        rolling_inter_mean = float(np.mean(self._inter_mean_history))
        baseline_elev = float(
            (rolling_inter_mean - self.baseline_inter_mean_median)
            / (self.baseline_inter_mean_std + 1e-6)
        )

        raw = np.array(
            [[mnf, inter_mean, inter_frac, mean_flow, inter_std, flow_trend, baseline_elev]],
            dtype=np.float32,
        )

        # Scale using fitted scaler
        scaled = self.if_scaler.transform(raw)

        # Clip to prevent extreme values
        clipped = np.clip(scaled, -self.clip_bound, self.clip_bound)

        return clipped, {
            "inter_mean": inter_mean,
            "flow_trend": flow_trend,
            "baseline_elev": baseline_elev,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main update
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, window):
        """
        Update detector with a new window of flow data (20 minutes).

        Combines CUSUM and Isolation Forest scores, applies persistence filter.
        Returns a dictionary with anomaly status and detailed scores.

        Args:
            window: Array of 20 flow values (1 per minute)

        Returns:
            Dictionary with keys:
            - anomaly:      Boolean, final decision after persistence filter
            - final_score:  Weighted fusion score [0, 1]
            - level2:       CUSUM results
            - level3:       Isolation Forest results
        """
        window = np.asarray(window, dtype=np.float32)

        # ── Level 2: CUSUM ────────────────────────────────────────────────────
        s_final, cusum_triggered = self._run_cusum(window)
        cusum_score = min(1.0, s_final / self.cusum_h)

        # ── Level 3: Isolation Forest ─────────────────────────────────────────
        features, feat_debug = self._extract_features(window)
        raw_if_score = float(self.if_model.decision_function(features)[0])

        if_triggered = raw_if_score < self.if_threshold

        # Normalise IF score to [0, 1]
        anomaly_distance = max(0.0, self.if_threshold - raw_if_score)
        if_score = min(1.0, anomaly_distance / max(self.if_score_scale, 1e-6))

        # ── Fusion ────────────────────────────────────────────────────────────
        final_score = self.w2 * cusum_score + self.w3 * if_score

        candidate_anomaly = final_score > self.decision_threshold

        # ── Persistence filter ────────────────────────────────────────────────
        if candidate_anomaly:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = 0

        final_anomaly = self._anomaly_streak >= self.persistence_windows

        return {
            "anomaly": bool(final_anomaly),
            "final_score": float(final_score),
            "level2": {
                "triggered": bool(cusum_triggered),
                "score": float(cusum_score),
            },
            "level3": {
                "triggered": bool(if_triggered),
                "score": float(if_score),
                "reconstruction_error": float(raw_if_score),   # kept for UI compat
                "flow_trend": float(feat_debug["flow_trend"]),
                "baseline_elev": float(feat_debug["baseline_elev"]),
            },
        }

    def reset(self):
        """Reset detector's internal state (CUSUM, persistence streak, rolling history)."""
        self.cusum_s = 0.0
        self._prev_appliance = False
        self._anomaly_streak = 0
        self._inter_mean_history.clear()
