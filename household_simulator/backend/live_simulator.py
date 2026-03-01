import json
import numpy as np

MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60


class LiveWaterFlowGenerator:
    def __init__(
        self,
        priors_path,
        daily_min_l=110,
        daily_max_l=170,
        max_flow_lpm=15.0,
        seed=42,
    ):
        np.random.seed(seed)

        with open(priors_path) as f:
            self.priors = json.load(f)["appliances"]

        self.DAILY_MIN_L = daily_min_l
        self.DAILY_MAX_L = daily_max_l
        self.MAX_FLOW_LPM = max_flow_lpm

        self.current_day = 0
        self.current_minute = 0
        self.day_flow = None

        # EXACT SAME single leak structure
        self.injected_leak = None

        self._generate_new_day()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def next(self):
        """
        Returns next minute's flow value (LPM)
        """

        if self.current_minute >= MINUTES_PER_DAY:
            self.current_day += 1
            self.current_minute = 0
            self._generate_new_day()

        flow_value = self.day_flow[self.current_minute] # type: ignore

        # EXACT ORIGINAL LEAK LOGIC
        if self.injected_leak is not None:
            global_min = self.global_minute()

            if (
                self.injected_leak["start"]
                <= global_min
                < self.injected_leak["end"]
            ):
                flow_value = self.injected_leak["flow_lpm"]

        self.current_minute += 1
        return float(flow_value)

    # --------------------------------------------------

    def inject_leak(self, duration_minutes=180, flow_lpm=0.4):
        start = self.global_minute()
        end = start + duration_minutes

        self.injected_leak = {
            "start": start,
            "end": end,
            "flow_lpm": flow_lpm,
        }

    def clear_leak(self):
        self.injected_leak = None

    # ==========================================================
    # INTERNAL (UNCHANGED)
    # ==========================================================

    def global_minute(self):
        return self.current_day * MINUTES_PER_DAY + self.current_minute

    def _generate_new_day(self):
        day_events = []

        for appliance in self.priors:
            day_events.extend(
                self._generate_events_for_day(appliance, self.current_day)
            )

        day_flow = self._render_day(day_events)

        # Daily volume scaling (UNCHANGED)
        day_volume = day_flow.sum()
        target_volume = np.random.uniform(
            self.DAILY_MIN_L,
            self.DAILY_MAX_L
        )

        scale = target_volume / max(day_volume, 1e-6)
        day_flow *= scale

        self.day_flow = day_flow

    # ==========================================================
    # EXACT ORIGINAL EVENT LOGIC
    # ==========================================================

    def _generate_events_for_day(self, appliance, day):
        name = appliance["appliance"]
        lam = appliance["activation"]["events_per_day"]["lambda"]
        hour_probs = appliance["timing"]["start_hour"]["p"]

        if name == "shower":
            n_events = 1
        elif name in ["toilet", "bidet"]:
            n_events = max(2, np.random.poisson(lam))
        elif name == "washingmachine":
            n_events = 1 if day % 7 == 0 else 0
        elif name == "dishwasher30":
            n_events = 1 if day % 7 == 0 else 0
        else:
            n_events = np.random.poisson(lam)

        events = []

        for _ in range(n_events):
            start_min = (
                np.random.choice(24, p=hour_probs) * 60
                + np.random.randint(60)
            )

            dur_cfg = appliance["duration"]

            if dur_cfg["type"] == "fixed":
                duration_s = dur_cfg["value"]
            else:
                duration_s = np.random.lognormal(
                    mean=np.log(dur_cfg["scale"]),
                    sigma=dur_cfg["shape"],
                )

            duration_s = np.clip(
                duration_s,
                appliance["constraints"]["min_duration_s"],
                appliance["constraints"]["max_duration_s"],
            )

            duration_min = int(np.ceil(duration_s / SECONDS_PER_MIN))

            if name in ["washingmachine", "dishwasher30"]:
                total_volume_L = np.random.uniform(40, 65)
                mean_flow_ml_s = (total_volume_L * 1000) / duration_s
            else:
                flow_cfg = appliance["flow"]["mean_flow"]
                mean_flow_ml_s = np.random.lognormal(
                    mean=np.log(flow_cfg["scale"]),
                    sigma=flow_cfg["shape"],
                )

            lpm = mean_flow_ml_s * 60 / 1000

            events.append({
                "start_min": int(start_min),
                "duration_min": duration_min,
                "flow_lpm": lpm,
            })

        return events

    def _render_day(self, events):
        flow = np.zeros(MINUTES_PER_DAY)

        for ev in events:
            start = ev["start_min"]
            dur = ev["duration_min"]
            lpm = ev["flow_lpm"]

            end = min(start + dur, MINUTES_PER_DAY)

            flow[start:end] += lpm
            flow[start:end] = np.minimum(
                flow[start:end],
                self.MAX_FLOW_LPM
            )

        return flow