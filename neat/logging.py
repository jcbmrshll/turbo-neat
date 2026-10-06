from typing import Any, Callable, Dict, Optional

from monitor import Monitor


def log_generation(
    monitor: Monitor,
    prev_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    generation_num: int,
    episode_data: Optional[Any] = None,
    episode_fn: Optional[Callable] = None,
) -> None:
    """Log one generation's metrics, with the test episode if there is one (packed by
    episode_fn into an Episode for the monitor server to render)."""
    metrics = {**prev_stats, **metrics}
    if episode_data is not None and episode_fn is not None:
        metrics["episode"] = episode_fn(episode_data)
    monitor.log(generation_num, metrics)
