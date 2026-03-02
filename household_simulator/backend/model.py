import numpy as np
from collections import deque


class HybridWaterAnomalyDetector:
    def __init__(self,
                 model,
                 flow_mean,
                 flow_std,
                 ae_threshold,
                 window=60,
                 rolling_window=20,
                 leak_threshold=0.4,
                 persistence_minutes=20,
                 std_delta_threshold=0.3,
                 w2=0.4,
                 w3=0.6,
                 decision_threshold=0.7):

        # -------- ML --------
        self.model = model
        self.flow_mean = flow_mean          # shape (2,)
        self.flow_std = flow_std            # shape (2,)
        self.ae_threshold = ae_threshold
        self.window = window

        # -------- Statistical --------
        self.rolling_window = rolling_window
        self.leak_threshold = leak_threshold
        self.persistence_minutes = persistence_minutes
        self.std_delta_threshold = std_delta_threshold

        # -------- Fusion --------
        self.w2 = w2
        self.w3 = w3
        self.decision_threshold = decision_threshold

        # -------- Buffers --------
        self.buffer = deque(maxlen=window)
        self.raw_flow_buffer = deque(maxlen=rolling_window)
        self.delta_buffer = deque(maxlen=rolling_window)

        self.prev_flow = None
        self.consecutive_leak_count = 0

    # ==========================================================
    # UPDATE (LIVE STREAMING)
    # ==========================================================

    def update(self, flow):

        # --------------------------------------------------
        # Compute derivative
        # --------------------------------------------------
        if self.prev_flow is None:
            delta = 0.0
        else:
            delta = flow - self.prev_flow

        self.prev_flow = flow

        # --------------------------------------------------
        # Update statistical buffers
        # --------------------------------------------------
        self.raw_flow_buffer.append(flow)
        self.delta_buffer.append(delta)

        # --------------------------------------------------
        # Level 2: Statistical Detection
        # --------------------------------------------------
        level2_score = 0.0
        level2_triggered = False

        if len(self.raw_flow_buffer) == self.rolling_window:

            mean_val = np.mean(self.raw_flow_buffer)
            std_delta = np.std(self.delta_buffer)

            if mean_val > self.leak_threshold and std_delta < self.std_delta_threshold:

                self.consecutive_leak_count += 1

                if self.consecutive_leak_count >= self.persistence_minutes:
                    level2_triggered = True

                    level2_score = min(
                        1.0,
                        (mean_val / self.leak_threshold) *
                        (1 - std_delta / self.std_delta_threshold)
                    )
            else:
                self.consecutive_leak_count = 0

        # --------------------------------------------------
        # Prepare ML 2-channel sample
        # --------------------------------------------------
        sample = np.array([flow, delta], dtype=np.float32)
        sample_scaled = (sample - self.flow_mean) / self.flow_std

        self.buffer.append(sample_scaled)

        level3_score = 0.0
        level3_triggered = False
        reconstruction_error = 0.0

        # --------------------------------------------------
        # Level 3: CNN Autoencoder
        # --------------------------------------------------
        if len(self.buffer) == self.window:

            window_array = np.array(self.buffer, dtype=np.float32)
            window_array = np.expand_dims(window_array, axis=0)

            reconstruction = self.model.predict(window_array, verbose=0)

            reconstruction_error = np.mean(
                (reconstruction - window_array) ** 2
            )

            level3_triggered = reconstruction_error > self.ae_threshold
            level3_score = min(
                1.0,
                reconstruction_error / self.ae_threshold
            )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------
        final_score = (self.w2 * level2_score) + (self.w3 * level3_score)
        final_anomaly = final_score > self.decision_threshold

        return {
            "anomaly": bool(final_anomaly),
            "final_score": float(final_score),

            "level2": {
                "triggered": bool(level2_triggered),
                "score": float(level2_score)
            },

            "level3": {
                "triggered": bool(level3_triggered),
                "score": float(level3_score),
                "reconstruction_error": float(reconstruction_error)
            }
        }
        
    def reset(self):
        self.buffer.clear()
        self.raw_flow_buffer.clear()
        self.delta_buffer.clear()
        self.prev_flow = None
        self.consecutive_leak_count = 0