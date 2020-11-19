import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback
from datetime import datetime

# Set up CPU or GPU
config = tf.ConfigProto(device_count={'CPU': 2, 'GPU': 2})
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
backend.set_session(sess)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class AutoencoderRNN():
    def __init__(self, n_units=20, drop_rate=0., state_size=3, action_size=1, steps=200):
        # Extract features from an RNN (LSTM)

        self.adim = action_size
        self.sdim = state_size
        self.n_steps = steps

        n_in = steps
        n_out = n_in - 1
        dim = state_size + action_size

        # define encoder
        visible = Input(shape=(n_in, dim))
        encoder = CuDNNLSTM(n_units)(visible)

        # define reconstruct decoder
        decoder1 = RepeatVector(n_in)(encoder)
        decoder1 = CuDNNLSTM(n_units, return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(dim))(decoder1)

        # define predict decoder
        decoder2 = RepeatVector(n_out)(encoder)
        decoder2 = CuDNNLSTM(n_units, return_sequences=True)(decoder2)
        decoder2 = TimeDistributed(Dense(dim))(decoder2)

        # tie it together
        # encoder_noisy = Dropout(rate=drop_rate)(encoder, training=True)
        self.latent = Model(inputs=visible, outputs=encoder)
        self.autoencoder = Model(inputs=visible, outputs=[decoder1, decoder2])
        self.autoencoder.compile(optimizer='rmsprop', loss='mse')


    def train(self, x_train, epochs=100, batch_size=10, verbose=1, pad_seq=True):
        # fit model
        lossHistory = LossHistory()
        start_time = datetime.now()

        print("Training model....")

        seq_in = x_train
        seq_out = x_train[:, 1:, :]

        self.autoencoder.fit(seq_in,
                             [seq_in, seq_out],
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=verbose,
                             callbacks=[lossHistory])
        end_time = datetime.now()
        print('')
        print("*********************************  End  *********************************")
        print()
        print("Training Finished, final loss is {}".format(lossHistory.losses[-1]))
        print('Duration: {}'.format(end_time - start_time))

        # Plot training loss
        plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)
        plt.show()

        return lossHistory

    def reconstruct(self, x_test):
        # demonstrate prediction
        return self.autoencoder.predict(x_test, verbose=0)


if __name__ == "__main__":
    from src.data.pendulum_data_generator import PendulumDataGenerator
    import os

    assets_path = "../../assets/data/"
    datafile = "pendulum_1000_samples_20190523-154923.pkl"

    policy_file_pendulum = os.path.join("/home/rafaelpossas/dev/bayes_sim/", "src", "models", "controllers", "PPO",
                                        "Pendulum-v0.pkl")

    g_pendulum = PendulumDataGenerator(policy_file=policy_file_pendulum, sufficient_stats="State-Action",
                                       assets_path="../../assets/data/", load_from_file=True, filename=datafile)

    params_pendulum, seq_in_pendulum = g_pendulum.gen(n_samples=None)

    rnn_pendulum = AutoencoderRNN()

    loss_history = rnn_pendulum.train(seq_in_pendulum, epochs=1000, batch_size=500, verbose=1)

    test_params_cartpole, x_test_cartpole = g_pendulum.gen(50, save=False)

    plot_dim = 0
    test_pt = 32
    y_plot_true = x_test_cartpole[test_pt, :, :]
    yhat = rnn_pendulum.reconstruct(x_test_cartpole)
    y_plot_pred = yhat[0][test_pt, :, :]
    x_plot = range(len(y_plot_pred[:, plot_dim]))
    fig = plt.figure(0)
    plt.plot(x_plot, y_plot_pred[:, plot_dim], '-b', label=r'Predicted')
    plt.plot(x_plot, y_plot_true[:len(y_plot_pred[:, plot_dim]), plot_dim], '-r', label=r'True')
    plt.legend(fontsize=10)
    plt.show()
