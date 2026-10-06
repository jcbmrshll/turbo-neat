# neat-jax
**neat-jax** is a hardware-accelerated implementation of the 2002 paper [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) (NeuroEvolution of Augmenting Toplogies) in JAX.

(cool visualization of evolving neural networks)

Inspired by compute constraints faced at that time, NEAT is an evolutionary training algorithm that evolves a population of tiny neural networking by perturbing weights, changing activations, and adding+removing neurons.

## Quickstart

## Monitoring runs
Training runs push metrics, episode renders and the champion's network to a small local dashboard:

```sh
uv run turbo-neat-monitor            # http://localhost:8008, runs stored in ./runs
uv run examples/boids.py --monitor   # in another shell
```

## Contributing

## Further Reading
- *Neural Network Evolution Playground with Backprop NEAT*, David Ha: https://blog.otoro.net/2016/05/07/backprop-neat/
- *Evolving Neural Networks through
Augmenting Topologies* (NEAT Paper): http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
