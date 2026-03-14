"""
Hybrid Water Anomaly Detection Model for Apartment Buildings

DESCRIPTION:
    Implements a hybrid anomaly detector for building-level water flow data, combining
    statistical CUSUM change detection with machine learning Isolation Forest.
    Designed for 50-apartment building aggregate flow monitoring.

DESIGN NOTES:
    - 7 features: mnf, inter_mean, inter_frac, mean_flow, inter_std, flow_trend, baseline_elev
      * flow_trend: linear regression slope of 20-min window → catches rising baselines
      * baseline_elev: normalised deviation of inter-appliance mean from training median
    - Rolling 60-min baseline tracker for live inference of baseline_elev
    - CUSUM k = 3.0: sits at ~67th percentile of normal inter-appliance flow (median 2.39 L/min,
      std 1.39), so natural variance cannot accumulate to h within a realistic window
    - CUSUM h = 15.0: requires sustained above-k flow for many minutes; normal appliance cycling
      provides frequent partial resets (s *= 0.5) that prevent drift on clean data
    - Appliance resets are partial (s *= 0.5, not s = 0): preserves accumulated leak evidence
      when an appliance starts on top of an active leak
    - CUSUM bypass: cusum_triggered=True raises candidate_anomaly regardless of fusion score
      because w_cusum=0.35 alone cannot cross decision_threshold via weighted fusion
    - IF bypass: if_triggered=True also raises candidate_anomaly independently, because at
      aggressive if_threshold (near 0) partial IF scores do not cross the fusion gate
    - persistence_windows=4: requires 4 consecutive candidate-anomaly minutes before alarm fires;
      primary false-positive guard when IF threshold is set close to zero

CALIBRATION (loaded from artifacts/calibration_building.json at server startup):
    cusum_k                    = 3.0   (67th pct of normal inter-appliance flow)
    cusum_h                    = 8.0
    if_threshold               = -0.02 (aggressive; catches 2-5 L/min leaks)
    if_score_scale             = 0.08
    decision_threshold         = 0.40
    persistence_windows        = 4
    baseline_inter_mean_median = 2.391 L/min (from training data)
    baseline_inter_mean_std    = 1.394 L/min

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
        # CUSUM — tuned to building inter-appliance baseline (~2.39 L/min)
        cusum_k=3.0,               # Must be above normal baseline; 3.0 = ~67th pct of normal inter-appliance flow
        cusum_h=8.0,               # High enough that normal variance can't reach it without a real leak
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
        w2=0.35,                   # CUSUM weight
        w3=0.65,                   # IF weight
        decision_threshold=0.40,   # Fusion-path gate; CUSUM/IF bypass rules also apply
        persistence_windows=4,     # Consecutive candidate windows before alarm fires
    ):
        """
        Initialize the hybrid anomaly detector.

        All parameters are loaded from artifacts/calibration_building.json at server
        startup; the defaults here reflect the current calibrated values.

        Args:
            if_model:                     Trained Isolation Forest model (7 features, 300 trees)
            if_scaler:                    StandardScaler fitted on 7 training features
            cusum_k:                      CUSUM reference level.  MUST be above the normal
                                          inter-appliance baseline (~2.39 L/min) to prevent
                                          accumulation on clean data.  Default 2.6.
            cusum_h:                      CUSUM trigger threshold (accumulated slack).  Default 8.0.
            noise_floor:                  Flow below this treated as zero (0.2 L/min)
            if_threshold:                 IF decision function cut-off.  Anomaly when raw score < this.
                                          Set aggressively close to 0 (-0.02) for 2–5 L/min detection.
            if_score_scale:               Range over which IF score transitions 0→1. Default 0.08.
            appliance_flow_thresh:        Flow >= this = appliance event; excluded from CUSUM
                                          inter-appliance accumulation.  Default 8.0 L/min.
            clip_bound:                   Clips scaled feature vector to [-clip, +clip] before IF
            baseline_inter_mean_median:   Training-set median of inter-appliance mean flow.
                                          Used for baseline_elev feature.  Default from calib JSON.
            baseline_inter_mean_std:      Training-set std dev of inter-appliance mean flow.
            w2, w3:                       CUSUM and IF weights in fusion (0.35 / 0.65)
            decision_threshold:           Fused score gate for anomaly declaration (0.40).
                                          Either CUSUM or IF can also bypass this gate independently.
            persistence_windows:          Consecutive candidate-anomaly windows required before
                                          alarm fires (4).  Primary false-positive guard.
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

        Design notes:
        - No zero-flow decay: in a 50-apartment building true zero flow is rare;
          decay would prevent accumulation of sustained-leak signal.
        - Appliance resets are PARTIAL (s *= 0.5), not full reset (s = 0).
          This preserves accumulated evidence of a small baseline leak even
          when appliances are running on top of it.
        - k = 3.0 sits above the normal inter-appliance baseline so CUSUM only
          accumulates when a real leak pushes flow above the reference level.
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

        # Both CUSUM and IF can independently bypass the weighted fusion threshold:
        # - w_cusum=0.35 alone can never cross decision_threshold=0.40 via fusion
        # - IF at partial confidence (raw score just below threshold) contributes
        #   too little via 0.65 × small_if_score to cross decision_threshold
        # Persistence filter (windows=4) is the guard against false positives.
        candidate_anomaly = (
            final_score > self.decision_threshold
            or cusum_triggered
            or if_triggered
        )

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
