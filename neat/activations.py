from typing import Callable, List

import jax
import jax.numpy as jnp

ActivationSelector = Callable[[jax.Array, jax.Array], jax.Array]
ActivationFn = Callable[[jax.Array], jax.Array]


def make_activation_selector_fn(activation_fns: List[ActivationFn]):
    return lambda i, x: jax.lax.switch(i, activation_fns, x)


def iden(x: jax.Array) -> jax.Array:
    return x


def inv(x: jax.Array) -> jax.Array:
    return 1 / (x + 1e-8)


relu = jax.nn.relu
sigmoid = jax.nn.sigmoid
tanh = jax.nn.tanh
absl = jnp.abs
sin = jnp.sin
cos = jnp.cos
exp = jnp.exp
