import math

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax
from tqdm import tqdm
# import torch

import equinox as eqx

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, *, key):
    t = jnp.linspace(0, 2 * math.pi, 16)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)
    print(x.shape, y.shape)
    return x, y

    # from tensorflow import keras
    # mnist = keras.datasets.mnist

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_train = jnp.array(x_train, dtype=jnp.float32)
    # x_test = jnp.array(x_test, dtype=jnp.float32)
    # y_train = jnp.array(y_train, dtype=jnp.float32)
    # y_test = jnp.array(y_test, dtype=jnp.float32)

    # sample, sample_label = x_train[0], y_train[0]
    # unit_1 = 10
    # unit_2 = 20
    # unit_3 = 30

    # i1 = 32
    # i2 = 64
    # i3 = 32
    # batch_size = 64
    # num_batches = 10
    # timestep = 50
    # import numpy as np
    # input_1_data = np.random.random((batch_size * num_batches, timestep, i1))
    # input_2_data = np.random.random((batch_size * num_batches, timestep, i2, i3))
    # target_1_data = np.random.random((batch_size * num_batches, unit_1))
    # target_2_data = np.random.random((batch_size * num_batches, unit_2, unit_3))
    # input_data = [input_1_data, input_2_data]
    # target_data = [target_1_data, target_2_data]
    # input_data = jnp.array(input_data)
    # target_data = jnp.array(target_data)
    # input_data = jax.device_put(input_data)
    # target_data = jax.device_put(target_data)

    # # return x_train, y_train
    # return input_data, target_data

# def get_data(dataset_size, *, key):
#     # (10000, 16, 2) (10000, 1)
#     T = 1000  # Generate a total of 1000 points
#     time = torch.arange(1, T + 1, dtype=torch.float32)
#     x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
#     tau = 4
#     features = torch.zeros((T - tau, tau))
#     for i in range(tau):
#         features[:, i] = x[i: T - tau + i]
#     labels = x[tau:].reshape((-1, 1))
#     features,labels = features[:dataset_size].cpu().detach().numpy(), labels[:dataset_size].cpu().detach().numpy()
#     features,labels = jnp.array(features), jnp.array(labels)
#     print(features.shape, labels.shape)
#     return features, labels
from equinox.module import Module, static_field
from equinox.custom_types import *
import jax.nn as jnn
import math
from typing import Optional

class GRUCellNew(Module):
    """A single step of a Gated Recurrent Unit (GRU).

    !!! example

        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: GRUCell

            def __init__(self, ...):
                self.cell = GRUCell(...)

            def __call__(self, xs):
                scan_fn = lambda state, input: (cell(input, state), None)
                init_state = jnp.zeros(self.cell.hidden_size)
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    bias_n: Optional[Array]
    input_size: int = static_field()
    hidden_size: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: Optional["jax.random.PRNGKey"],
        **kwargs
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)

        ihkey, hhkey, bkey, bkey2 = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        self.weight_ih = jrandom.uniform(
            ihkey, (3 * hidden_size, input_size), minval=-lim, maxval=lim
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (3 * hidden_size, hidden_size), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(
                bkey, (3 * hidden_size,), minval=-lim, maxval=lim
            )
            self.bias_n = jrandom.uniform(
                bkey2, (hidden_size,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None
            self.bias_n = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(
        self, input: Array, hidden: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """
        if self.use_bias:
            bias = 0
            bias_n = 0
        else:
            bias = self.bias
            bias_n = self.bias_n
        igates = jnp.split(self.weight_ih @ input + bias, 3)
        hgates = jnp.split(self.weight_hh @ hidden, 3)
        reset = jnn.sigmoid(igates[0] + hgates[0])
        inp = jnn.sigmoid(igates[1] + hgates[1])
        new = jnn.tanh(igates[2] + reset * (hgates[2] + bias_n))
        return new + inp * (hidden - new)

class LSTMCell(Module):
    """A single step of a Long-Short Term Memory unit (LSTM).
    !!! example
        This is often used by wrapping it into a `jax.lax.scan`. For example:
        ```python
        class Model(Module):
            cell: LSTMCell
            def __init__(self, ...):
                self.cell = LSTMCell(...)
            def __call__(self, xs):
                scan_fn = lambda state, input: (cell(input, state), None)
                init_state = (jnp.zeros(self.cell.hidden_size),
                              jnp.zeros(self.cell.hidden_size))
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    input_size: int = static_field()
    hidden_size: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: "jax.random.PRNGKey",
        **kwargs
    ):
        """**Arguments:**
        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)

        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        self.weight_ih = jrandom.uniform(
            ihkey, (4 * hidden_size, input_size), minval=-lim, maxval=lim
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (4 * hidden_size, hidden_size), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(
                bkey, (4 * hidden_size,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(self, input, hidden, *, key=None):
        """**Arguments:**
        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a 2-tuple of JAX arrays, each of
            shape `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        The updated hidden state, which is a 2-tuple of JAX arrays, each of shape
        `(hidden_size,)`.
        """
        h, c = hidden
        lin = self.weight_ih @ input + self.weight_hh @ h
        if self.use_bias:
            lin = lin + self.bias
        i, f, g, o = jnp.split(lin, 4)
        i = jnn.sigmoid(i)
        f = jnn.sigmoid(f)
        g = jnn.tanh(g)
        o = jnn.sigmoid(o)
        c = f * c + i * g
        h = o * jnn.tanh(c)
        return (h, c)




class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    # linearin: eqx.nn.Linear
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey, lkeyin = jrandom.split(key,3)
        self.hidden_size = hidden_size
        self.cell = GRUCellNew(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        # self.linearin = eqx.nn.Linear(in_size, hidden_size, use_bias=False, key=lkeyin)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)

def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    hidden_size=10,
    depth=1,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, key=data_key)
    # xs, ys = get_data()
    dataset_size = ys.shape[0]
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        # return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
        return -jnp.mean(y * jnp.log(pred_y))

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in tqdm(zip(range(steps), iter_data), total = steps):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    print(f"num_correct={num_correct}")
    print(f"si={dataset_size}")
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")

main()
