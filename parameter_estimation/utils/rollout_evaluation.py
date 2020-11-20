import numpy as np


def set_state(env, qp, qv):
    env.set_state(qp, qv)

def evaluate_actions(env, start_sim_state, actions):
    """Evaluates a given policy in a particular environment, given actions
    """

    states = []
    next_states = []

    qp, qv = start_sim_state
    set_state(env, qp, qv)
    state = env._get_obs()

    for i, action in enumerate(actions):
        next_state, r, d, _ = env.step(action)
        
        states.append(state)
        next_states.append(next_state)

        state = next_state

    trajectory = np.concatenate([
                    states,
                    actions,
                    next_states
                ], axis=-1)
    
    return np.transpose(trajectory, (1, 0, 2))