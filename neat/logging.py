from typing import Any, Callable, Dict, Optional

from monitor import Episodes, Monitor
from neat.fitness import Recording


def log_generation(
    monitor: Monitor,
    prev_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    generation_num: int,
    episode_data: Optional[Any] = None,
    episode_fn: Optional[Callable] = None,
    recording: Optional[Recording] = None,
    genome_ids: Optional[Any] = None,
) -> None:
    """Log one generation's metrics, with the test episode if there is one (packed by
    episode_fn into an Episode for the monitor server to render), and the episodes
    the members played for their fitness, if they were recorded.

    episode_fn packs the recorded episodes too, all at once: it gets the frames with
    a leading episode axis, so it should only index and stack along trailing axes."""
    metrics = {**prev_stats, **metrics}
    if episode_data is not None and episode_fn is not None:
        metrics["episode"] = episode_fn(episode_data)
    if recording is not None and episode_fn is not None and genome_ids is not None:
        metrics["episodes"] = Episodes(
            episode_fn(recording.frames), players=genome_ids[recording.players]
        )
    monitor.log(generation_num, metrics)
