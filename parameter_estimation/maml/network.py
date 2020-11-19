from jax.experimental import stax # neural network library
from jax.experimental.stax import Dense, Relu

def generate_network(out_features):
    # Use stax to set up network initialization and evaluation functions
    net_init, net_apply = stax.serial(
        Dense(40), Relu,
        Dense(40), Relu,
        Dense(out_features)
    )

    return net_init, net_apply