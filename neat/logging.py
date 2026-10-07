from typing import Any, Callable, Dict, Optional

import numpy as np

from monitor import Episodes, Monitor
from neat.fitness import Recording
from neat.species import SpeciesPCA, describe_species_pca


def log_generation(
    monitor: Monitor,
    prev_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    generation_num: int,
    episode_data: Optional[Any] = None,
    episode_fn: Optional[Callable] = None,
    recording: Optional[Recording] = None,
    genome_ids: Optional[Any] = None,
    species_pca: Optional[SpeciesPCA] = None,
    episode_players: Optional[Any] = None,
) -> None:
    """Log one generation's metrics, with the test episode if there is one (packed by
    episode_fn into an Episode for the monitor server to render), and the episodes
    the members played for their fitness, if they were recorded.

    episode_fn packs the recorded episodes too, all at once: it gets the frames with
    a leading episode axis, so it should only index and stack along trailing axes.

    species_pca, the generation's species map, is still on the device: it's copied
    off here, on the logging thread."""
    metrics = {**prev_stats, **metrics}
    if episode_data is not None and episode_fn is not None:
        metrics["episode"] = episode = episode_fn(episode_data)
        # the genomes playing the episode, one per seat, for the monitor to name
        if episode_players is not None:
            episode.data["players"] = np.asarray(episode_players)
    if recording is not None and episode_fn is not None and genome_ids is not None:
        metrics["episodes"] = Episodes(
            episode_fn(recording.frames), players=genome_ids[recording.players]
        )
    if species_pca is not None:
        metrics["species_pca"] = describe_species_pca(species_pca)
    monitor.log(generation_num, metrics)
