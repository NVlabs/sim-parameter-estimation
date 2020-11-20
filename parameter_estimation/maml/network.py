from jax.experimental import stax # neural network library
from jax.experimental.stax import Dense, Relu, Sigmoid

def generate_network(out_features, hidden_size):
    # Use stax to set up network initialization and evaluation functions
    net_init, net_apply = stax.serial(
        Dense(hidden_size), Relu,
        Dense(hidden_size), Relu,
        Dense(out_features), Sigmoid
    )

    return net_init, net_apply