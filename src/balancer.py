"""Balancer: a PID controller that nudges each page's mix so what the user actually reads matches their sliders."""
import numpy as np


class PIDBalancer:
    """Cruise control for the topic mix.

    After every page it compares the target mix with the mix of articles the user has clicked so far:
    error = target - clicked share. The next page's mix is the target plus a correction:
      P: kp x error now
      I: ki x error added up over the session (a gap that lasts gets a stronger push)
      D: kd x change in error since the last page (damps overreaction)
    max_push caps how far any bucket can be moved from the target: the "how hard to push" knob for when a user
    keeps skipping a bucket they asked for.

    Optional feedforward (a head start): `start_mix` is a page mix guessed before the session from the user's past
    clicks, e.g. from `feedforward_mix`. The PID then corrects around it instead of around the raw target.

    Used as a simulator policy: balancer(page, clicked_counts, targets) -> page mix.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.0, max_push: float = 0.2, prior_clicks: float = 5.0,
                 start_mix: np.ndarray | None = None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_push = max_push
        self.prior_clicks = prior_clicks  # pretend the session starts with this many clicks exactly on target (calms early noise)
        self.start_mix = start_mix

    def __call__(self, page: int, clicked_counts: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if page == 0:
            self.integral = np.zeros_like(targets)
            self.previous = np.zeros_like(targets)

        clicked_share = (clicked_counts + self.prior_clicks * targets) / (clicked_counts.sum(1, keepdims=True) + self.prior_clicks)
        error = targets - clicked_share
        self.integral += error
        correction = self.kp * error + self.ki * self.integral + self.kd * (error - self.previous)
        self.previous = error

        base = targets if self.start_mix is None else self.start_mix
        mix = base + np.clip(correction, -self.max_push, self.max_push)
        mix = np.clip(mix, 0, None)
        return mix / mix.sum(1, keepdims=True)


def feedforward_mix(targets: np.ndarray, click_chance: np.ndarray) -> np.ndarray:
    """Head start: show each bucket in proportion to target / expected click chance, so the expected reading mix
    matches the target. A bucket you click half as often gets shown twice as much."""
    mix = targets / click_chance
    return mix / mix.sum(1, keepdims=True)
