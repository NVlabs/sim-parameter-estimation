import tensorflow as tf
from src.utils.util import store_args, create_fully_connected_nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dim_observation, dim_goal, dim_action, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.observation_tensor = inputs_tf['observation']
        self.goal_tensor = inputs_tf['goal']
        self.action_tensor = inputs_tf['action']

        # Prepare inputs for actor and critic.
        observation = self.o_stats.normalize(self.observation_tensor)
        goal = self.g_stats.normalize(self.goal_tensor)
        # Policy Input uses UFVA concept, where a we extend our state space by concatenating the goal
        # More Info On: http://proceedings.mlr.press/v37/schaul15.pdf
        input_pi = tf.concat(axis=1, values=[observation, goal])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            # Creates a fully connected Neural Net with 'n' layers of same size and last layer with action_size
            # the function returns the tensorflow output tensor
            self.pi_tf = self.max_u * tf.tanh(
                create_fully_connected_nn(input_pi, [self.hidden] * self.layers + [self.dim_action]))

        with tf.variable_scope('Q'):
            # for policy training
            # Same Idea of UFVA applies in here, however, the Q function goes from Q(s,a) to Q(s,g,a)
            input_Q = tf.concat(axis=1, values=[observation, goal, self.pi_tf / self.max_u])
            self.Q_pi_tf = create_fully_connected_nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[observation, goal, self.action_tensor / self.max_u])
            # This network shares the weights with the target network hence why we have reuse=True
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = create_fully_connected_nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
