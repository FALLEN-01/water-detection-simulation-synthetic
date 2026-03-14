"""
Hybrid Water Anomaly Detection Model for Apartment Buildings

DESCRIPTION:
    Implements a hybrid anomaly detector for building-level water flow data, combining
    statistical CUSUM change detection with machine learning Isolation Forest.
    Designed for 50-apartment building aggregate flow monitoring.

KEY FEATURES:
    - CUSUM (level 2): Detects persistent low-flow deviations (leaks)
    - Isolation Forest (level 3): Detects statistical anomalies in feature space
    - Feature extraction: 5 features per window (mnf, inter_mean, inter_frac, mean_flow, inter_std)
    - Persistence filter: Reduces false positives by requiring consecutive anomalies
    - Building-scale thresholds: Calibrated for 50-apartment aggregate

DEPENDENCIES:
    - numpy: Data processing
    - sklearn: Isolation Forest model and scaler (passed in from training script)
"""

import numpy as np


class HybridWaterAnomalyDetector:
    """Hybrid anomaly detector combining CUSUM and Isolation Forest for building-level detection."""

    def __init__(
        self,
        if_model,
        if_scaler,
        cusum_k=0.5,           # Building scale: ~10x household (was 0.01)
        cusum_h=20.0,          # Building scale: ~10x household (was 2.0)
        noise_floor=0.2,       # Building scale: ~10x household (was 0.02)
        if_threshold=-0.05,    # IF model decision threshold (model-dependent)
        if_score_scale=0.1,    # IF score scaling (model-dependent)
        appliance_flow_thresh=8.0,     # Building scale: ~10x household (was 0.8)
        clip_bound=10.0,       # Clipping for feature scaling
        w2=0.4,                # CUSUM weight
        w3=0.6,                # IF weight
        decision_threshold=0.65,
        persistence_windows=2,
    ):
        """
        Initialize the hybrid anomaly detector.

        Args:
            if_model: Trained Isolation Forest model
            if_scaler: StandardScaler fitted on training features
            cusum_k: CUSUM sensitivity parameter (threshold for low flow)
            cusum_h: CUSUM trigger threshold
            noise_floor: Flow threshold below which noise is filtered
            if_threshold: Isolation Forest decision function threshold
            if_score_scale: Scale factor for IF score normalization
            appliance_flow_thresh: Flow threshold distinguishing appliance vs baseline flow
            clip_bound: Clipping bound for scaled features
            w2, w3: Weights for CUSUM and IF in fusion
            decision_threshold: Threshold for declaring anomaly after fusion
            persistence_windows: Number of consecutive anomalous windows before triggering
        """
        self.if_model = if_model
        self.if_scaler = if_scaler

        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.noise_floor = noise_floor

        self.if_threshold = if_threshold
        self.if_score_scale = if_score_scale
        self.clip_bound = clip_bound

        self.appliance_flow_thresh = appliance_flow_thresh

        self.w2 = w2
        self.w3 = w3
        self.decision_threshold = decision_threshold
        self.persistence_windows = persistence_windows

        # State tracking
        self._anomaly_streak = 0
        self.cusum_s = 0.0
        self._prev_appliance = False

    def _run_cusum(self, window):
        """
        Run CUSUM change detection on a window of flow data.
        Returns the final CUSUM statistic and whether a change was triggered.

        CUSUM detects sustained deviations in the flow baseline.
        """
        s = self.cusum_s
        triggered = False
        prev = self._prev_appliance

        for lpm in window:
            # Distinguish appliance (high flow) vs baseline (low flow)
            appliance = lpm >= self.appliance_flow_thresh

            # Reset CUSUM on appliance start
            if appliance and not prev:
                s = 0.0

            # Accumulate for baseline flow
            if not appliance:
                if lpm <= self.noise_floor:
                    # Decay signal on very low flow (sensor noise)
                    s *= 0.8
                else:
                    # Accumulate if flow > noise floor but < appliance threshold
                    delta = lpm - self.cusum_k
                    s = max(0.0, s + delta)

                # Trigger if accumulation exceeds threshold
                if s >= self.cusum_h:
                    triggered = True

            prev = appliance

        self.cusum_s = s
        self._prev_appliance = prev

        return s, triggered

    def _extract_features(self, window):
        """
        Extract 5 statistical features from a window for Isolation Forest.

        Features:
        1. mnf: 10th percentile of non-zero flows (baseline minimum)
        2. inter_mean: Mean of inter-appliance flows (baseline average)
        3. inter_frac: Fraction of inter-appliance periods
        4. mean_flow: Average window flow
        5. inter_std: Std dev of inter-appliance flows

        Returns scaled feature vector (1, 5) ready for IF model.
        """
        inter = window[window < self.appliance_flow_thresh]
        nonzero = window[window > 0.0]

        # 10th percentile of nonzero flows
        mnf = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.0

        # Mean of inter-appliance (baseline) flows
        inter_mean = float(inter.mean()) if len(inter) > 0 else 0.0

        # Fraction of samples in inter-appliance range
        inter_frac = float((inter > self.noise_floor).mean()) if len(inter) > 0 else 0.0

        # Overall mean
        mean_flow = float(window.mean())

        # Std of inter-appliance flows
        inter_std = float(inter.std()) if len(inter) > 0 else 0.0

        # Assemble feature vector
        raw = np.array(
            [[mnf, inter_mean, inter_frac, mean_flow, inter_std]],
            dtype=np.float32,
        )

        # Scale using fitted scaler
        scaled = self.if_scaler.transform(raw)

        # Clip to prevent extreme values
        clipped = np.clip(scaled, -self.clip_bound, self.clip_bound)

        return clipped

    def update(self, window):
        """
        Update detector with a new window of flow data (20 minutes).

        Combines CUSUM and Isolation Forest scores, applies persistence filter.
        Returns a dictionary with anomaly status and detailed scores.

        Args:
            window: Array of 20 flow values (1 per minute)

        Returns:
            Dictionary with keys:
            - anomaly: Boolean, final decision after persistence filter
            - final_score: Weighted fusion score [0, 1]
            - level2: CUSUM detector results
            - level3: Isolation Forest detector results
        """
        window = np.asarray(window, dtype=np.float32)

        # ─────────────────────────────────────────────────────────
        # Level 2: CUSUM (Persistent Low-Flow Detection)
        # ─────────────────────────────────────────────────────────

        s_final, cusum_triggered = self._run_cusum(window)
        cusum_score = min(1.0, s_final / self.cusum_h)

        # ─────────────────────────────────────────────────────────
        # Level 3: Isolation Forest (Statistical Anomaly)
        # ─────────────────────────────────────────────────────────

        features = self._extract_features(window)
        raw_if_score = float(self.if_model.decision_function(features)[0])

        if_triggered = raw_if_score < self.if_threshold

        # Normalize IF score to [0, 1]
        anomaly_distance = max(0.0, self.if_threshold - raw_if_score)
        if_score = min(1.0, anomaly_distance / self.if_score_scale)

        # ─────────────────────────────────────────────────────────
        # Fusion: Weighted combination
        # ─────────────────────────────────────────────────────────

        final_score = self.w2 * cusum_score + self.w3 * if_score

        candidate_anomaly = final_score > self.decision_threshold

        # ─────────────────────────────────────────────────────────
        # Persistence Filter: Require consecutive anomalous windows
        # ─────────────────────────────────────────────────────────

        if candidate_anomaly:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = 0

        final_anomaly = self._anomaly_streak >= self.persistence_windows

        # ─────────────────────────────────────────────────────────
        # Return Results
        # ─────────────────────────────────────────────────────────

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
                "reconstruction_error": float(raw_if_score),
            },
        }

    def reset(self):
        """Reset detector's internal state (CUSUM, persistence streak)."""
        self.cusum_s = 0.0
        self._prev_appliance = False
        self._anomaly_streak = 0
