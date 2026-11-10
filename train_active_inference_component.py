import numpy as np

def active_inference_step(prior, likelihood, obs_index, utility, alpha=0.1):
    """
    Performs one step of Active Inference mathematically.
    prior: 1D array, prior over states
    likelihood: 2D array, P(Observation | State)
    obs_index: int, index of the current observation
    utility: 1D array, log prior over outcomes (goals/preferences)
    alpha: float, learning rate for updating the prior
    """
    # 1. Perception (State Inference)
    # Bayesian update of beliefs about states given observation
    obs_likelihood = likelihood[obs_index, :]
    posterior_states = prior * obs_likelihood
    posterior_states /= np.sum(posterior_states) + 1e-12

    # 2. Planning (Action Selection via Expected Free Energy)
    # Expected Free Energy (G) for each action (simplified version)
    # In a full POMDP, G would evaluate policies. Here we evaluate states directly for simplicity

    # Ambiguity: Expected entropy of observations given states
    # Risk: KL divergence between predicted observations and preferred observations

    # Simulate action choices by predicting future states
    # For this minimal test, let's assume actions directly correspond to transitioning to a target state
    num_states = len(prior)
    expected_free_energy = np.zeros(num_states)

    # Predictive distribution over observations for each possible next state
    for s in range(num_states):
        pred_obs = likelihood[:, s]

        # Risk (KL Divergence): D_KL( P(o|s) || P_pref(o) )
        risk = np.sum(pred_obs * (np.log(pred_obs + 1e-12) - utility))

        # Ambiguity: H(o|s)
        ambiguity = -np.sum(pred_obs * np.log(pred_obs + 1e-12))

        expected_free_energy[s] = risk + ambiguity

    # Action selection (Softmax over negative Expected Free Energy)
    action_probs = np.exp(-expected_free_energy)
    action_probs /= np.sum(action_probs)

    # Update prior for next step based on posterior
    next_prior = (1 - alpha) * prior + alpha * posterior_states
    next_prior /= np.sum(next_prior)

    return posterior_states, expected_free_energy, action_probs, next_prior

def test_active_inference():
    np.random.seed(42)
    print("Testing Active Inference Component...")

    num_states = 3
    num_obs = 3

    # Prior over states
    prior = np.ones(num_states) / num_states

    # Likelihood P(Obs | State). Mostly diagonal
    likelihood = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])

    # Utility (Preferences) over observations. Prefer observation 2
    utility = np.array([-2.0, -2.0, 0.0]) # Log probabilities

    obs_index = 0 # Current observation

    posterior_states, efe, action_probs, next_prior = active_inference_step(
        prior, likelihood, obs_index, utility
    )

    print(f"Initial Prior: {prior}")
    print(f"Observation index: {obs_index}")
    print(f"Utility (Log Pref): {utility}")
    print(f"Posterior States: {posterior_states}")
    print(f"Expected Free Energy: {efe}")
    print(f"Action Probabilities: {action_probs}")
    print(f"Next Prior: {next_prior}")

    # Basic assertions
    assert np.allclose(np.sum(posterior_states), 1.0)
    assert np.allclose(np.sum(action_probs), 1.0)
    assert np.allclose(np.sum(next_prior), 1.0)

    # Because observation is 0, posterior should favor state 0
    assert np.argmax(posterior_states) == 0

    # Because utility strongly favors observation 2 (which correlates with state 2),
    # EFE should be lowest (best) for state 2, and action probs should favor state 2
    assert np.argmin(efe) == 2
    assert np.argmax(action_probs) == 2

    print("Active Inference Component test passed successfully.")

if __name__ == '__main__':
    test_active_inference()
