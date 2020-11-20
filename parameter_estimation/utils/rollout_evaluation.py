import numpy as np


def evaluate_actions(env, actions):
    """Evaluates a given policy in a particular environment, given actions
    """

    states = []
    next_states = []

    # TODO - Look at eval code
    env.seed(1)
    state = env.reset()

    for action in actions:
        action_expanded = np.repeat(action[:, np.newaxis], env.nenvs, axis=1).T 
        next_state, r, d, _ = env.step(action_expanded)
        
        states.append(state)
        next_states.append(next_state)

        state = next_state

    nagents = np.array(states).shape[1]
    actions = np.expand_dims(actions, 1)
    actions = np.repeat(actions, nagents, axis=1)

    trajectory = np.concatenate([
                    states,
                    actions,
                    next_states
                ], axis=-1)

    return np.transpose(trajectory, (1, 0, 2))