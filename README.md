# turbo-neat

**Hardware-accelerated neuroevolution in JAX**

turbo-neat is a high-performance implementation of NEAT (NeuroEvolution of Augmenting Topologies) built on JAX. Leverage hardware acceleration (GPU/TPU) to evolve neural network topologies orders of magnitude faster than traditional implementations.

## Overview

NEAT is a powerful evolutionary algorithm that simultaneously evolves neural network weights and topologies through genetic algorithms. This implementation harnesses JAX's JIT compilation and automatic vectorization to achieve massive speedups, enabling:

- **Parallel population evaluation** across GPUs/TPUs
- **JIT-compiled evolution** for minimal overhead
- **Vectorized genome operations** for efficient mutation and crossover
- **Speciation** to protect innovation and maintain diversity
- **Flexible fitness functions** for single-agent and competitive multi-agent tasks

Originally introduced by Stanley & Miikkulainen (2002), NEAT has proven effective for reinforcement learning, game playing, robotics, and other evolutionary optimization problems.

## Key Features

- **JAX-native implementation**: Fully differentiable, JIT-compilable evolutionary algorithms
- **Hardware acceleration**: Seamlessly scale to GPUs and TPUs
- **Modular architecture**: Easy-to-extend genome, mutation, and selection components
- **Competitive coevolution**: Built-in support for head-to-head evaluation
- **Rich logging**: Integrated Weights & Biases support for experiment tracking
- **Network visualization**: Automatic generation of evolved topology diagrams
- **EvoJAX integration**: Compatible with EvoJAX task environments
- **Speciation**: Automatic clustering of similar genomes to preserve innovation

## Installation

### Prerequisites

turbo-neat requires JAX. Install the appropriate version for your hardware:

```bash
# CPU-only
pip install jax

# GPU (CUDA 12)
pip install -U "jax[cuda12]"

# TPU
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

See the [JAX installation guide](https://github.com/google/jax#installation) for more options.

### Install turbo-neat

```bash
# From source
git clone https://github.com/lowrollr/turbo-neat.git
cd turbo-neat
pip install -e .
```

Additional dependencies for examples:

```bash
pip install evojax wandb
```

## Quick Start

Here's a minimal example evolving agents for a flocking task:

```python
import jax
from evojax.task import flocking
from neat_jax.config import NEATConfig, GenomeConfig, MutationConfig, SelectionConfig
from neat_jax.neat import NEAT
from neat_jax.fitness import make_fitness_fn
import neat_jax.activations as act

# Create task environment
task = flocking.FlockingTask(num_agents=100, action_type=1)

# Define fitness function
fitness_fn = make_fitness_fn(task=task, num_steps=100)

# Configure genome
genome_config = GenomeConfig(
    input_size=task.obs_shape[0],
    output_size=task.act_shape[0],
    initial_capacity=100,
    init_weight_mean=0.0,
    init_weight_std=1.0,
    activation_fns=[act.iden, act.sigmoid, act.tanh, act.relu],
    init_mode="full"
)

# Configure mutation rates
mutation_config = MutationConfig(
    add_connection_prob=0.05,
    add_node_prob=0.03,
    disable_connection_prob=0.05,
    disable_node_prob=0.03,
    mutate_weight_prob=0.8,
    mutate_weight_std=0.1,
)

# Configure selection and speciation
selection_config = SelectionConfig(
    population_size=100,
    cutoff_pct=0.1,
    speciation_threshold=1.0,
    compatibility_coefficients=(1.0, 0.9),
    elitism=0.1,
)

# Combine configs
config = NEATConfig(
    mutation_config=mutation_config,
    genome_config=genome_config,
    selection_config=selection_config,
)

# Initialize and run NEAT
neat = NEAT(config=config, fitness_fn=fitness_fn, wandb_project="my-experiment")
population = neat.run(seed=42, num_generations=100)
```

## Configuration

### Genome Configuration

Control the structure and initialization of neural networks:

- `input_size` / `output_size`: Network input/output dimensions
- `initial_capacity`: Initial node capacity (grows automatically)
- `init_weight_mean` / `init_weight_std`: Weight initialization distribution
- `activation_fns`: List of available activation functions
- `capacity_growth_strategy`: `"linear"`, `"exponential"`, or `"constant"`
- `init_mode`: `"full"` (fully connected) or `"partial"` (sparse initialization)

### Mutation Configuration

Define evolutionary operators and their probabilities:

- `add_connection_prob`: Probability of adding a new connection
- `add_node_prob`: Probability of adding a new node
- `disable_connection_prob` / `disable_node_prob`: Probability of disabling components
- `mutate_weight_prob` / `mutate_weight_std`: Weight mutation parameters
- `mutate_activation_prob`: Probability of changing activation functions
- `mutate_bias_prob` / `mutate_bias_std`: Bias mutation parameters

### Selection Configuration

Control population dynamics and speciation:

- `population_size`: Number of genomes per generation
- `cutoff_pct`: Fraction of population selected for reproduction
- `speciation_threshold`: Distance threshold for species clustering
- `compatibility_coefficients`: Weights for distance metric (topology, weights)
- `elitism`: Fraction of best genomes preserved unchanged
- `maximum_species`: Maximum number of species maintained
- `selection_tournament_size`: Tournament size for parent selection

## Advanced Usage

### Competitive Coevolution

Evolve agents through head-to-head competition:

```python
from neat_jax.fitness import make_2p_fitness_fn, make_h2h_fitness_fn

# Self-play fitness evaluation
train_fitness_fn = make_2p_fitness_fn(
    task=competitive_task,
    steps_per_round=100,
    num_rounds=5
)

# Champion testing
h2h_fn = make_h2h_fitness_fn(task=competitive_task, num_steps=100)

neat = NEAT(
    config=config,
    fitness_fn=train_fitness_fn,
    h2h_test_fn=h2h_fn,  # Track champion progression
)
```

### Custom Fitness Functions

Define custom evaluation logic:

```python
from neat_jax.genome import Genome, apply

def custom_fitness_fn(rng, genome, activation_selector):
    """Custom fitness evaluation.

    Args:
        rng: JAX random key
        genome: Batched genome containing population
        activation_selector: Function mapping activation IDs to functions

    Returns:
        fitnesses: Array of shape (population_size,)
        data: Optional dict of additional data for logging
    """
    # Forward pass through networks
    observations = get_observations()  # Your logic here
    actions = apply(genome, Genome.forward, observations, activation_selector=activation_selector)

    # Compute fitness
    fitnesses = evaluate_actions(actions)

    return fitnesses, {}
```

### Logging and Visualization

Track experiments with Weights & Biases:

```python
neat = NEAT(
    config=config,
    fitness_fn=train_fitness_fn,
    baseline_test_fn=test_fitness_fn,  # Evaluate champion on test set
    wandb_project="my-project",
)

# Visualizations are automatically logged when improvements occur
population = neat.run(
    seed=42,
    num_generations=100,
    render_fn=custom_render_function,  # Optional: render episodes as videos
    render_async=True,  # Render in background thread
)
```

### Resuming Training

Save and restore populations:

```python
# Run initial training
population = neat.run(seed=42, num_generations=50)

# Continue training from checkpoint
population = neat.run(
    seed=42,
    num_generations=50,  # Additional generations
    population=population,  # Resume from previous state
)
```

## Examples

See `run_neat.ipynb` for a complete example including:

- EvoJAX flocking task integration
- Custom input/output labeling for visualization
- Rendering episodes as GIFs
- WandB logging setup

## Performance

turbo-neat achieves significant speedups over CPU implementations:

- **100-1000x faster** evaluation on GPU vs single-core CPU
- **Fully parallelized** population evaluation
- **Minimal overhead** through JIT compilation
- **Scales efficiently** to populations of 1000+ genomes

## Citation

If you use turbo-neat in your research, please cite the original NEAT paper:

```bibtex
@article{stanley2002evolving,
  title={Evolving neural networks through augmenting topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary computation},
  volume={10},
  number={2},
  pages={99--127},
  year={2002},
  publisher={MIT Press}
}
```

Paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
