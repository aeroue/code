# -*- coding: utf-8 -*-
"""
Rewritten code based on aif_functions_isobeliefs_convergent.py
Focusing on Rendezvous Task using Active Inference (choice_heuristic) only.
Includes fixes for belief update, EFE calculation, softmax, KL divergence,
random initial beliefs, and integrated academic plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
# from palettable.colorbrewer.qualitative import Set1_9 # Using cm.tab10 instead
import copy
import time
import itertools
import os
import pandas as pd # Keep for potential data analysis later if needed

#------------------------------------------------------------------------------------------
# ----------------- Math & Utility Functions (Corrected Softmax/KL) ----------------------
#------------------------------------------------------------------------------------------
def softmax(x):
    """Compute softmax values for each sets of scores in x.
       Returns 1D array if input is 1D, else 2D array if input is 2D.
    """
    input_is_1d = (x.ndim == 1)
    if input_is_1d:
        x = x.reshape(1, -1)

    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-32)

    if input_is_1d:
        return probabilities.flatten()
    else:
        return probabilities
# def softmax(x):
#     """Compute softmax values. Returns 1D if input is 1D."""
#     if np.isscalar(x): return np.array([1.0]) # Handle scalar input
#     input_is_1d = False
#     if x.ndim == 1:
#         input_is_1d = True
#         x = x.reshape(1, -1)
#     if x.size == 0: return np.array([])

#     max_val = np.max(x, axis=1, keepdims=True)
#     # Subtracting max makes exp more stable (avoids large positive numbers)
#     # Clip ensures no extreme negative numbers after subtraction either
#     stable_x = np.clip(x - max_val, -700, None) # Clip before exp
#     e_x = np.exp(stable_x)
#     probabilities = e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-32)

#     if input_is_1d: return probabilities.flatten()
#     else: return probabilities

def log_stable(x):
    """Compute log values safely."""
    return np.log(np.maximum(x, 1e-32)) # Avoid log(0)

def wrapToPi(x):
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

def compute_distances(x, goals):
    """Compute Euclidean distances between agents (x) and goals."""
    if x.ndim == 1: x = x.reshape(1, -1)
    if goals.ndim == 1: goals = goals.reshape(1, -1)
    if x.shape[1] < 2 or goals.shape[1] < 2 or x.size == 0 or goals.size == 0:
        # Return empty array or handle error appropriately
        return np.empty((x.shape[0], goals.shape[0])) * np.nan # Indicate failure

    diff = x[:, np.newaxis, :2] - goals[np.newaxis, :, :2]
    distances = np.linalg.norm(diff, axis=2)
    return distances

def generate_spread_out_goals(num_goals, env_size, min_distance):
    """Generate goal locations reasonably spread out."""
    goals = []
    max_attempts = num_goals * 100
    attempts = 0
    while len(goals) < num_goals and attempts < max_attempts:
        new_goal = np.random.rand(2) * env_size
        attempts += 1
        if not goals or all(np.linalg.norm(new_goal - np.array(goal)) >= min_distance for goal in goals):
            goals.append(new_goal.tolist())
    if len(goals) < num_goals:
        print(f"Warning: Only generated {len(goals)} goals out of {num_goals} requested.")
    return np.array(goals)

#------------------------------------------------------------------------------------------
# ----------------- State, Observation & Prediction (Unchanged) --------------------------
#------------------------------------------------------------------------------------------

def simulate_observation(true_position, observation_error_std=0.5, sim_type='A', observed_agent_id=None):
    # (Code from previous version - assumed correct)
    observation = {'position': np.zeros((2,))-1e6, 'heading': float(-1e6), 'type': sim_type, 'observed_agent': observed_agent_id}
    error_std_pos = observation_error_std
    error_std_head = observation_error_std * 0.1
    if sim_type == 's':
        error_std_pos *= 0.1
        error_std_head *= 0.1
    observation['position'] = true_position[:2] + np.random.normal(0, error_std_pos, 2)
    observation['heading'] = true_position[2] + np.random.normal(0, error_std_head)
    observation['heading'] = wrapToPi(observation['heading'])
    return observation

def predict_agent_position(agent_position, velocity, heading_change, dt):
    # (Code from previous version - assumed correct)
    agent_prediction = np.copy(agent_position)
    new_heading = wrapToPi(agent_position[2] + heading_change)
    agent_prediction[0] += velocity * np.cos(new_heading) * dt
    agent_prediction[1] += velocity * np.sin(new_heading) * dt
    agent_prediction[2] = new_heading
    return agent_prediction

def parse_observations(observations):
    # (Code from previous version - assumed correct)
    num_agents = len(observations)
    if num_agents == 0: return np.array([])
    agent_states = np.zeros((num_agents, 3))
    for idx, obs in enumerate(observations):
        # Basic check if observation is valid
        if isinstance(obs, dict) and 'position' in obs and 'heading' in obs:
             agent_states[idx, :2] = obs['position']
             agent_states[idx, 2] = obs['heading']
        else:
             print(f"Warning: Invalid observation format at index {idx}: {obs}")
             # Handle invalid observation, e.g., fill with NaN or default
             agent_states[idx, :] = np.nan

    return agent_states.copy()

#------------------------------------------------------------------------------------------
# ---- Likelihoods, Belief Update & Epistemic Reasoning (Revised) ------------------------
#------------------------------------------------------------------------------------------

def custom_cdist(observer_agent_id, agent_poses, goals, observer_type, all_agent_types, eta=30.0): # Increased default eta
    """Compute evidence based on observer's sensor type."""
    # (Code largely from previous version, using increased default eta)
    num_agents = agent_poses.shape[0]
    num_goals = goals.shape[0]
    if num_agents == 0 or num_goals == 0: return np.zeros((0,0))

    evidence = np.ones((num_agents, num_goals), dtype=float) / num_goals # Start with uniform

    agent_xy = agent_poses[:, :2]
    goals_xy = goals[:, :2]

    # 1. Distance evidence
    distances_to_goals = compute_distances(agent_xy, goals_xy)
    if np.any(np.isnan(distances_to_goals)): return np.ones((num_agents, num_goals))/num_goals # Handle distance error
    max_dist_val = np.max(distances_to_goals) if distances_to_goals.size > 0 else 1.0
    if max_dist_val < 1e-6: max_dist_val = 1.0
    # Higher eta makes evidence more sensitive to distance changes
    distance_evidence = np.exp(-eta * distances_to_goals / max_dist_val)
    distance_evidence_softmax = softmax(distance_evidence)

    # 2. Alignment evidence
    vec_agent_to_goal = goals_xy[np.newaxis, :, :] - agent_xy[:, np.newaxis, :]
    norm_vec_agent_to_goal = vec_agent_to_goal / (np.linalg.norm(vec_agent_to_goal, axis=2, keepdims=True) + 1e-6)
    vec_observer_to_agent = agent_xy[np.newaxis, :, :] - agent_xy[observer_agent_id, np.newaxis, :]
    norm_vec_observer_to_agent = vec_observer_to_agent / (np.linalg.norm(vec_observer_to_agent, axis=2, keepdims=True) + 1e-6)
    alignment_evidence_softmax = np.zeros((num_agents, num_goals))
    for i in range(num_agents):
        if i == observer_agent_id: continue
        cosine_similarity = np.dot(norm_vec_agent_to_goal[i, :, :], norm_vec_observer_to_agent[0, i, :])
        # Higher eta makes softmax more sensitive to alignment
        alignment_evidence_softmax[i, :] = softmax(eta * cosine_similarity)

    # 3. Assign evidence based on types
    for i in range(num_agents):
        agent_type = all_agent_types[i]
        dist_valid = distance_evidence_softmax.shape == (num_agents, num_goals)
        align_valid = alignment_evidence_softmax.shape == (num_agents, num_goals)
        if observer_type in ['s', 'A']:
             if agent_type in ['s', 'A'] and dist_valid: evidence[i, :] = distance_evidence_softmax[i, :]
             elif agent_type == 'B' and dist_valid: evidence[i, :] = distance_evidence_softmax[i, :]
        elif observer_type == 'B':
             if agent_type in ['s', 'A'] and align_valid: evidence[i, :] = alignment_evidence_softmax[i, :]
             elif agent_type == 'B' and align_valid: evidence[i, :] = alignment_evidence_softmax[i, :]

    if observer_type == 's' and dist_valid:
         evidence[observer_agent_id, :] = distance_evidence_softmax[observer_agent_id, :]

    return np.squeeze(evidence) # Shape (agents, goals)

def calculate_likelihood_for_config_rendezvous(agent_evidence, target_goal_idx):
    """Calculate likelihood P(Obs | Config=target_goal_idx)."""
    # Config for rendezvous: all agents target target_goal_idx
    num_agents, num_goals = agent_evidence.shape
    if not (0 <= target_goal_idx < num_goals): return 0.0 # Safety check
    # Likelihood is product of evidence for each agent towards the *target* goal
    log_likelihood = np.sum(log_stable(agent_evidence[:, target_goal_idx]))
    return np.exp(np.clip(log_likelihood, -700, 700)) # Clip exp

# def calculate_goal_likelihood_distribution(observer_agent_id, agent_poses, goals, observer_type, all_agent_types):
#     """Calculate the likelihood distribution P(Obs | Goal=g) for each goal g."""
#     num_agents = agent_poses.shape[0]
#     num_goals = goals.shape[0]
#     if num_goals == 0: return np.array([])

#     # Get evidence P(Goal=g | Obs_from_agent_i) for each agent i
#     agent_evidence = custom_cdist(observer_agent_id, agent_poses, goals, observer_type, all_agent_types)
#     if agent_evidence.size == 0 or agent_evidence.shape != (num_agents, num_goals):
#         print(f"Warning: custom_cdist returned invalid shape {agent_evidence.shape}. Using uniform likelihood.")
#         return np.ones(num_goals) / num_goals

#     # Calculate likelihood P(Obs | Goal=g) for each goal g
#     goal_likelihoods = np.zeros(num_goals, dtype=np.float64)
#     for g_idx in range(num_goals):
#         # Assumes P(Obs | Goal=g) is proportional to product of P(Goal=g | Obs_i) across agents i
#         goal_likelihoods[g_idx] = calculate_likelihood_for_config_rendezvous(agent_evidence, g_idx)

#     # Normalize likelihoods to form a distribution (or keep unnormalized for Bayes?)
#     # For Bayesian update P(G|O) propto P(O|G)*P(G), we need P(O|G)
#     # Let's return the unnormalized likelihoods, normalization happens in Bayes update
#     # Add small constant if all are zero to avoid issues
#     if np.sum(goal_likelihoods) < 1e-9:
#        goal_likelihoods += 1e-9

#     return goal_likelihoods # Shape (num_goals,) - Represents P(Obs | Goal=g)
def calculate_likelihood_distribution_rigorous(observer_agent_id, agent_poses, goals, observer_type, all_agent_types):
    """
    Calculate the likelihood distribution P(Obs | Goal=g) for each rendezvous goal g rigorously.
    P(Obs | Goal=g) = P(Obs | Config=(g,g,...,g)) = Product_i [Evidence(Agent_i targets g)]
    """
    num_agents = agent_poses.shape[0]
    num_goals = goals.shape[0]
    if num_agents == 0 or num_goals == 0: return np.array([])

    # 1. Get agent evidence: evidence[i, j] = Evidence that agent i targets goal j
    agent_evidence = custom_cdist(observer_agent_id, agent_poses, goals, observer_type, all_agent_types)
    if agent_evidence.size == 0 or agent_evidence.shape != (num_agents, num_goals):
        print(f"Warning (rigorous): custom_cdist returned invalid shape {agent_evidence.shape}. Using uniform likelihood.")
        return np.ones(num_goals) / num_goals if num_goals > 0 else np.array([])

    # Ensure evidence is numerically stable for log calculation
    stable_agent_evidence = np.maximum(agent_evidence, 1e-32)

    # 2. Calculate likelihood P(Obs | Goal=g) for each rendezvous goal g
    goal_likelihoods = np.zeros(num_goals, dtype=np.float64)
    for target_goal_idx in range(num_goals):
        # Likelihood for config (g, g, ..., g) is the product of evidence for each agent targeting g
        # Product_i [Evidence(Agent_i targets g)]
        # In log space: Sum_i log[Evidence(Agent_i targets g)]
        log_likelihood_sum = np.sum(np.log(stable_agent_evidence[:, target_goal_idx]))

        # Convert back to likelihood: P(Obs | Goal=g) = exp(log_likelihood_sum)
        goal_likelihoods[target_goal_idx] = np.exp(np.clip(log_likelihood_sum, -700, 700)) # Clip before exp

    # 3. Return the unnormalized likelihoods for the rendezvous configurations
    # Add small constant if all are zero to avoid issues in Bayesian update
    if np.sum(goal_likelihoods) < 1e-9:
       goal_likelihoods += 1e-9

    # Shape (num_goals,) - Represents P(Obs | Goal=g), needed for 1D belief update
    return goal_likelihoods

def compute_consensus_likelihood(likelihoods_all_perspectives, method='certainty_weighted'):
    """Compute consensus likelihood using a specified method.

    Args:
        likelihoods_all_perspectives (np.ndarray): Shape (num_perspectives, num_goals).
        method (str): Method to use:
            'geometric_mean' (default): Simple geometric mean (equal weights).
            'arithmetic_mean': Simple arithmetic mean.
            'certainty_weighted': Certainty-weighted geometric mean (inverse entropy).

    Returns:
        np.ndarray: Consensus likelihood distribution (unnormalized), shape (num_goals,).
    """
    if likelihoods_all_perspectives.ndim == 1: return likelihoods_all_perspectives
    num_perspectives, num_goals = likelihoods_all_perspectives.shape
    if num_perspectives == 0: return np.array([])
    if num_perspectives == 1: return likelihoods_all_perspectives.flatten()

    # --- Pre-process for numerical stability (needed for methods using log) ---
    valid_likelihoods = np.maximum(likelihoods_all_perspectives, 1e-32)
    valid_likelihoods = np.nan_to_num(valid_likelihoods, nan=1e-32, posinf=1e30, neginf=1e-32)

    # --- Select Method ---
    if method == 'High Entropy Weighted Arithmetic Mean':
        # Normalize each perspective's likelihood row for entropy calculation (needed to calc weights)
        row_sums = valid_likelihoods.sum(axis=1, keepdims=True)
        normalized_likelihoods = np.divide(valid_likelihoods, row_sums,
                                           out=np.ones_like(valid_likelihoods) / num_goals if num_goals > 0 else np.zeros_like(valid_likelihoods),
                                           where=row_sums > 1e-9)
        # Calculate weights (clipped entropy) - THIS IS THE ORIGINAL AIF METHOD
        # Note: Using the valid_likelihoods (potentially unnormalized) for the entropy calculation here,
        # as the original code seemed to apply entropy to the direct likelihood rows.
        # This is slightly ambiguous in the original, let's stick to entropy of the probability distribution.
        entropies = np.array([calculate_shannon_entropy(p) for p in normalized_likelihoods])
        consensus_weights = np.clip(entropies, 0, 20) # Use entropy directly as weight (clipped)

        # Weighted arithmetic mean (using valid_likelihoods)
        consensus_likelihood_sum = np.zeros(num_goals, dtype=float)
        total_weight = np.sum(consensus_weights) # Need to know total weight for potential normalization if desired

        if total_weight < 1e-9:
            # If weights are all zero (e.g., all entropies clipped to 0), fall back to simple arithmetic mean
             consensus_likelihood = np.mean(valid_likelihoods, axis=0)
             # print("Warning: AIF Original Consensus weights near zero, falling back to arithmetic mean.")
        else:
            for idx in range(num_perspectives):
                # Original AIF used weighted sum: likelihood * weight
                consensus_likelihood_sum += valid_likelihoods[idx, :] * consensus_weights[idx]

            # --- IMPORTANT DIFFERENCE ---
            # The *original* aif code immediately applied SOFTMAX to this weighted sum.
            # To replicate that behavior faithfully:
            consensus_likelihood = softmax(consensus_likelihood_sum)
            # This means the output of this specific method *IS ALREADY NORMALIZED* unlike others.

    elif method == 'certainty_weighted':
        # Normalize each perspective's likelihood row for entropy calculation
        row_sums = valid_likelihoods.sum(axis=1, keepdims=True)
        normalized_likelihoods = np.divide(valid_likelihoods, row_sums,
                                           out=np.ones_like(valid_likelihoods) / num_goals if num_goals > 0 else np.zeros_like(valid_likelihoods),
                                           where=row_sums > 1e-9)
        # Calculate weights
        entropies = np.array([calculate_shannon_entropy(p) for p in normalized_likelihoods])
        weights = 1.0 / (entropies + 1e-6)
        total_weight = np.sum(weights)
        if total_weight < 1e-9:
            normalized_weights = np.ones(num_perspectives) / num_perspectives
        else:
            normalized_weights = weights / total_weight
        # Weighted geometric mean
        log_likelihoods = np.log(valid_likelihoods)
        weighted_mean_log_likelihood = np.sum(normalized_weights[:, np.newaxis] * log_likelihoods, axis=0)
        consensus_likelihood = np.exp(np.clip(weighted_mean_log_likelihood, -700, 700))

    elif method == 'geometric_mean': # Default
        log_likelihoods = np.log(valid_likelihoods)
        mean_log_likelihood = np.mean(log_likelihoods, axis=0)
        consensus_likelihood = np.exp(np.clip(mean_log_likelihood, -700, 700))

    else:
        print(f"Warning: Unknown consensus method '{method}'. Using geometric_mean.")
        log_likelihoods = np.log(valid_likelihoods)
        mean_log_likelihood = np.mean(log_likelihoods, axis=0)
        consensus_likelihood = np.exp(np.clip(mean_log_likelihood, -700, 700))


    # --- Final Check ---
    if np.sum(consensus_likelihood) < 1e-9:
       consensus_likelihood += 1e-9 # Add small constant if all are zero

    return consensus_likelihood # Shape (num_goals,)

def update_belief_bayesian(prior, likelihood):
    """Perform Bayesian update: posterior propto likelihood * prior."""
    if prior.size != likelihood.size:
        print(f"Error: Prior ({prior.shape}) and Likelihood ({likelihood.shape}) size mismatch.")
        return prior # Return old prior if sizes don't match

    unnormalized_posterior = likelihood * prior
    posterior_sum = np.sum(unnormalized_posterior)

    if posterior_sum > 1e-9:
        posterior = unnormalized_posterior / posterior_sum
    else:
        # print(f"Warning: Posterior sum near zero. Resetting to uniform or keeping prior?")
        # Resetting to uniform might be safer if likelihood is consistently zero
        num_configs = prior.size
        posterior = np.ones(num_configs) / num_configs if num_configs > 0 else np.array([])
        # posterior = prior # Alternative: Keep prior

    # Ensure valid probability distribution
    posterior = np.clip(posterior, 0, 1)
    final_sum = np.sum(posterior)
    if abs(final_sum - 1.0) > 1e-6 and final_sum > 1e-9:
        posterior /= final_sum

    return posterior

# def get_belief_update(robot_id, observations, goals, agent_vars, current_belief):
#     """
#     Calculates the likelihood P(Obs|Goal) and updates the belief P(Goal|Obs).
#     Handles first-order and epistemic (consensus likelihood) reasoning.
#     Returns the updated belief (posterior).
#     """
#     observed_poses = parse_observations(observations)
#     if observed_poses.size == 0: return current_belief

#     num_goals = len(goals)
#     num_agents = agent_vars['num_agents']
#     all_agent_types = agent_vars['agent_types']

#     if current_belief is None or current_belief.size != num_goals:
#         print(f"Warning: Agent {robot_id} invalid current belief. Resetting to uniform.")
#         current_belief = np.ones(num_goals) / num_goals if num_goals > 0 else np.array([])

#     if num_goals == 0: return current_belief # Cannot update if no goals

#     use_ep = agent_vars['use_ep']

#     if not use_ep:
#         # --- First-Order Reasoning ---
#         observer_type = all_agent_types[robot_id]
#         likelihood_dist = calculate_goal_likelihood_distribution(
#             robot_id, observed_poses, goals, observer_type, all_agent_types
#         )
#     else:
#         # --- Higher-Order Reasoning (Epistemic) ---
#         likelihoods_all_perspectives = np.zeros((num_agents, num_goals), dtype=np.float64)
#         for perspective_agent_id in range(num_agents):
#             perspective_agent_type = all_agent_types[perspective_agent_id]
#             single_likelihood = calculate_goal_likelihood_distribution(
#                 perspective_agent_id, observed_poses, goals, perspective_agent_type, all_agent_types
#             )
#             if single_likelihood.shape == (num_goals,):
#                 likelihoods_all_perspectives[perspective_agent_id, :] = single_likelihood
#             else:
#                  # Fallback to uniform if calculation fails for a perspective
#                  likelihoods_all_perspectives[perspective_agent_id, :] = np.ones(num_goals) / num_goals


#         likelihood_dist = compute_consensus_likelihood(likelihoods_all_perspectives)

#     # Bayesian Update
#     posterior_belief = update_belief_bayesian(current_belief, likelihood_dist)

#     return posterior_belief
def get_belief_update(robot_id, observations, goals, agent_vars, current_belief):
    """
    Calculates the likelihood P(Obs|Goal) using the rigorous method
    and updates the belief P(Goal|Obs). Handles first-order and epistemic reasoning.
    Returns the updated belief (posterior).
    """
    observed_poses = parse_observations(observations)
    if observed_poses.size == 0: return current_belief

    num_goals = len(goals)
    num_agents = agent_vars['num_agents']
    all_agent_types = agent_vars['agent_types']

    if current_belief is None or current_belief.size != num_goals:
        print(f"Warning: Agent {robot_id} invalid current belief. Resetting to uniform.")
        current_belief = np.ones(num_goals) / num_goals if num_goals > 0 else np.array([])

    if num_goals == 0: return current_belief # Cannot update if no goals

    use_ep = agent_vars['use_ep']

    if not use_ep:
        # --- First-Order Reasoning ---
        observer_type = all_agent_types[robot_id]
        # Use the rigorous likelihood calculation
        likelihood_dist = calculate_likelihood_distribution_rigorous(
            robot_id, observed_poses, goals, observer_type, all_agent_types
        )
    else:
        # --- Higher-Order Reasoning (Epistemic) ---
        likelihoods_all_perspectives = np.zeros((num_agents, num_goals), dtype=np.float64)
        for perspective_agent_id in range(num_agents):
            perspective_agent_type = all_agent_types[perspective_agent_id]
            # Each perspective uses the rigorous likelihood calculation
            single_likelihood = calculate_likelihood_distribution_rigorous(
                perspective_agent_id, observed_poses, goals, perspective_agent_type, all_agent_types
            )
            if single_likelihood.shape == (num_goals,):
                likelihoods_all_perspectives[perspective_agent_id, :] = single_likelihood
            else:
                 likelihoods_all_perspectives[perspective_agent_id, :] = np.ones(num_goals) / num_goals if num_goals > 0 else np.array([])

        # Compute consensus from these rigorous likelihoods
        likelihood_dist = compute_consensus_likelihood(likelihoods_all_perspectives, method='geometric_mean')

    # Bayesian Update remains the same
    posterior_belief = update_belief_bayesian(current_belief, likelihood_dist)

    return posterior_belief

#------------------------------------------------------------------------------------------
# ----------------- Active Inference & Decision Making (Corrected EFE) -------------------
#------------------------------------------------------------------------------------------
def calculate_kl_divergence(q, prior_q=None):
    """Calculate KL divergence KL(q || p) where p is delta function on max(q).
       Or KL(q || prior_q) if prior_q is provided."""
    if q.size <= 1: return 0.0

    q = np.maximum(q, 1e-32) # Ensure q is positive
    q_sum = np.sum(q)
    if abs(q_sum - 1.0) > 1e-5: # Ensure q is normalized
        if q_sum < 1e-9: return 0.0 # Cannot normalize zero vector
        q = q / q_sum

    if prior_q is not None:
        # Calculate KL(q || prior_q)
        if q.shape != prior_q.shape:
            print("Warning: Shape mismatch for KL(q || prior).")
            return 0.0 # Or some indicator of error
        prior_q = np.maximum(prior_q, 1e-32)
        prior_q_sum = np.sum(prior_q)
        if abs(prior_q_sum - 1.0) > 1e-5:
            if prior_q_sum < 1e-9: return 0.0
            prior_q = prior_q / prior_q_sum
        kl_div = np.sum(q * (np.log(q) - np.log(prior_q)))
    else:
        # Calculate KL(q || p) where p is delta function on max(q)
        # Simplified calculation: Entropy(p) - CrossEntropy(p, q) = 0 - (-sum(p*log(q))) = sum(p*log(q))
        # Since p is delta function, this is log(q[max_idx])
        # KL(p || q) is sum p * log(p/q) = p[max_idx]*log(p[max_idx]/q[max_idx]) = 1*log(1/q[max_idx]) = -log(q[max_idx])
        max_idx = np.argmax(q)
        kl_div = -log_stable(q[max_idx]) # Use stable log

    # No need to normalize KL divergence itself generally
    # Optional: Normalize by log(number of states) for value between 0 and 1
    # return kl_div / np.log(q.size) if q.size > 1 else 0.0
    return kl_div


def calculate_shannon_entropy(p):
    """Calculate Shannon entropy H(p) = -sum(p * log(p))."""
    if p.size == 0: return 0.0
    p = np.maximum(p, 1e-32) # Ensure positivity
    p_sum = np.sum(p)
    if abs(p_sum - 1.0) > 1e-5: # Ensure normalization
        if p_sum < 1e-9: return np.log(p.size) if p.size > 0 else 0.0 # Max entropy for zero vector
        p = p / p_sum
    return -np.sum(p * np.log(p))

def calculate_goal_reward(agent_id, posterior_belief, agent_pos, goals, agent_vars):
    """Calculate pragmatic value: negative normalized distance to most likely goal."""
    if posterior_belief is None or posterior_belief.size == 0: return 1.0 # Max penalty if no belief
    num_goals = len(goals)
    if posterior_belief.size != num_goals: return 1.0 # Penalty if belief size mismatch

    most_likely_goal_idx = np.argmax(posterior_belief)
    if not (0 <= most_likely_goal_idx < num_goals): return 1.0 # Invalid index

    target_goal_pos = goals[most_likely_goal_idx][:2]
    agent_xy = agent_pos[:2]
    distance_to_target = np.linalg.norm(agent_xy - target_goal_pos)

    # Normalize distance
    env_size = agent_vars.get('env_size', 1.0) # Get env_size safely
    max_dist = np.sqrt(2 * env_size**2) if env_size > 0 else 1.0
    normalized_distance = np.clip(distance_to_target / (max_dist + 1e-6), 0, 1)

    # Reward is negative distance (we minimize EFE, so cost is positive distance)
    pragmatic_cost = normalized_distance
    return pragmatic_cost


def calculate_expected_free_energy(predicted_posterior, current_prior, agent_id, predicted_agent_pos, goals, agent_vars):
    """Calculate EFE = Epistemic Value + Pragmatic Value."""
    if predicted_posterior is None or predicted_posterior.size == 0: return np.inf
    if abs(np.sum(predicted_posterior) - 1.0) > 1e-5: return np.inf # Check validity

    # 1. Epistemic Value = Uncertainty (Entropy) + Information Gain (KL Divergence)
    #    Information Gain KL(posterior || prior)
    #    Ambiguity/Uncertainty H(posterior)
    entropy = calculate_shannon_entropy(predicted_posterior)
    # KL divergence between predicted posterior and current belief (prior for this step)
    # Use current_prior passed correctly
    kl_div = calculate_kl_divergence(predicted_posterior, prior_q=current_prior)

    # 2. Pragmatic Value (Cost)
    pragmatic_cost = calculate_goal_reward(agent_id, predicted_posterior, predicted_agent_pos, goals, agent_vars)

    # Combine: Minimize EFE = Ambiguity + KL Divergence + Pragmatic Cost
    alpha = 1.0 # Weight epistemic (uncertainty + info gain)
    beta = 1.0  # Weight pragmatic cost
    # Note: Original paper often formulates EFE relative to a target distribution P,
    # EFE = DKL[Q(s|pi)||P(s|m)] + H[Q(o|pi)]. This simplifies under assumptions.
    # Here we use: Ambiguity + KL(posterior||prior) + PragmaticCost
    efe = alpha * (entropy + kl_div) + beta * pragmatic_cost

    if not np.isfinite(efe):
        # print(f"Warning: EFE non-finite. E={entropy}, KL={kl_div}, P={pragmatic_cost}. Post={predicted_posterior}, Prior={current_prior}")
        return np.inf
    return efe


def choice_heuristic(current_positions, current_beliefs, agent_params):
    """Find action minimizing EFE using Active Inference."""
    # (Largely similar structure to previous versions, calls corrected functions)
    velocity_options = agent_params['velocity_options']
    heading_options = agent_params['heading_options']
    goals = agent_params['goals']
    agent_id = agent_params['agent_id']
    dt = agent_params['dt']

    best_action_idx = -1
    min_efe = np.inf
    best_posterior_for_action = current_beliefs # Default
    best_velocity = velocity_options[len(velocity_options)//2] if velocity_options else 0.0
    best_heading_change = 0.0

    action_idx = 0
    if not hasattr(velocity_options, '__iter__'): velocity_options = [velocity_options]
    if not hasattr(heading_options, '__iter__'): heading_options = [heading_options]

    # --- Simulate observations for the *current* state (needed for likelihood calc) ---
    current_observations = []
    for idx in range(agent_params['num_agents']):
        obs_type = agent_params['agent_types'][agent_id] if idx != agent_id else 's'
        obs = simulate_observation(current_positions[idx],
                                   agent_params['observation_error_std'],
                                   sim_type=obs_type, observed_agent_id=idx)
        current_observations.append(obs)
    # --- End Current Observation Simulation ---

    for velocity in velocity_options:
        for heading_change in heading_options:
            predicted_self_pos = predict_agent_position(current_positions[agent_id], velocity, heading_change, dt)
            predicted_positions = np.copy(current_positions)
            predicted_positions[agent_id] = predicted_self_pos

            # Simulate observations based on PREDICTED state
            predicted_observations = []
            for idx in range(agent_params['num_agents']):
                 obs_type = agent_params['agent_types'][agent_id] if idx != agent_id else 's'
                 pred_obs = simulate_observation(predicted_positions[idx],
                                                 agent_params['observation_error_std'],
                                                 sim_type=obs_type, observed_agent_id=idx)
                 predicted_observations.append(pred_obs)

            # Predict posterior belief resulting from this action
            # Note: Pass current_beliefs as the prior for this prediction step
            predicted_posterior = get_belief_update(agent_id, predicted_observations, goals, agent_params, current_beliefs)

            if predicted_posterior is None or predicted_posterior.size == 0 or not np.all(np.isfinite(predicted_posterior)):
                 efe = np.inf
            else:
                 # Calculate EFE comparing predicted_posterior to current_beliefs (as prior)
                 efe = calculate_expected_free_energy(predicted_posterior, current_beliefs, agent_id, predicted_self_pos, goals, agent_params)

            if efe < min_efe:
                min_efe = efe
                best_action_idx = action_idx
                best_velocity = velocity
                best_heading_change = heading_change
                best_posterior_for_action = predicted_posterior # Store the posterior resulting from best action

            action_idx += 1

    if best_action_idx == -1: # Handle no valid action found
        print(f"Warning: Agent {agent_id} using default action (no EFE improvement).")
        # Keep default action, recalculate posterior for consistency
        predicted_self_pos = predict_agent_position(current_positions[agent_id], best_velocity, best_heading_change, dt)
        predicted_positions = np.copy(current_positions); predicted_positions[agent_id] = predicted_self_pos
        predicted_observations = []
        for idx in range(agent_params['num_agents']):
             obs_type = agent_params['agent_types'][agent_id] if idx != agent_id else 's'
             pred_obs = simulate_observation(predicted_positions[idx], agent_params['observation_error_std'], sim_type=obs_type, observed_agent_id=idx)
             predicted_observations.append(pred_obs)
        best_posterior_for_action = get_belief_update(agent_id, predicted_observations, goals, agent_params, current_beliefs)
        min_efe = 0.0 # Assign default EFE if none calculated

    # IMPORTANT: Return the posterior that results from taking the BEST action
    return best_velocity, best_heading_change, min_efe if np.isfinite(min_efe) else 0.0, best_posterior_for_action


def make_decision_active_inference(agent_vars):
    """Agent decision-making using choice_heuristic."""
    # (Largely unchanged structure, calls corrected choice_heuristic)
    agent_id = agent_vars['agent_id']
    agent_positions = agent_vars['agent_positions']
    current_beliefs = agent_vars['beliefs']

    if current_beliefs is None or current_beliefs.size == 0:
        print(f"Error: Agent {agent_id} invalid beliefs before decision. Defaulting.")
        # Provide default valid belief if needed for calculation fallback
        num_goals = len(agent_vars.get('goals', []))
        current_beliefs = np.ones(num_goals) / num_goals if num_goals > 0 else np.array([])
        if current_beliefs.size == 0:
             return (0.0, 0.0), current_beliefs, np.inf # Cannot recover

    # choice_heuristic now calculates the posterior resulting from the best action
    best_velocity, best_heading_change, best_efe_score, next_belief = choice_heuristic(
        agent_positions, current_beliefs, agent_vars # Removed observations arg, calc inside heuristic
    )

    best_action = (best_velocity, best_heading_change)

    if next_belief is None or next_belief.size == 0:
        # print(f"Warning: Agent {agent_id} decision resulted in invalid next_belief. Keeping current.")
        next_belief = current_beliefs # Fallback

    # Returns the action to take and the belief state *after* taking that action
    return best_action, next_belief, best_efe_score

def greedy_decision(agent_id, agent_poses, goals, agent_vars):
    """Simple greedy decision: move towards the goal with highest own evidence."""
    observer_type = agent_vars['agent_types'][agent_id]
    all_agent_types = agent_vars['agent_types']
    num_goals = len(goals)
    if num_goals == 0: return (0.0, 0.0) # No goals

    # Get evidence ONLY from this agent's perspective
    agent_evidence = custom_cdist(agent_id, agent_poses, goals, observer_type, all_agent_types)

    # Check if evidence calculation was successful and shape is right
    if agent_evidence.size == 0 or agent_evidence.shape[0] <= agent_id or agent_evidence.shape[1] != num_goals:
        print(f"Warning (Greedy): Agent {agent_id} failed to get valid evidence. Staying still.")
        return (0.0, 0.0)

    # Find the goal index with the maximum evidence for this agent
    self_evidence = agent_evidence[agent_id, :]
    target_goal_idx = np.argmax(self_evidence)

    # --- Move towards the target goal ---
    current_pos = agent_poses[agent_id]
    target_pos = goals[target_goal_idx][:2]
    vector_to_goal = target_pos - current_pos[:2]
    distance_to_goal = np.linalg.norm(vector_to_goal)

    # Choose max velocity (or scale based on distance?) - Simple approach: max velocity
    velocity_options = agent_vars['velocity_options']
    best_velocity = velocity_options[-1] if velocity_options else 0.0
    # Cap velocity if close to goal to avoid overshooting? Optional refinement.
    # max_step = best_velocity * agent_vars['dt']
    # if distance_to_goal < max_step:
    #     best_velocity = distance_to_goal / agent_vars['dt']

    # Calculate desired heading change
    angle_to_goal = np.arctan2(vector_to_goal[1], vector_to_goal[0])
    heading_diff = wrapToPi(angle_to_goal - current_pos[2])

    # Choose heading change closest to desired change within options
    heading_options = agent_vars['heading_options']
    if heading_options:
        best_heading_change_idx = np.argmin(np.abs(np.array(heading_options) - heading_diff))
        best_heading_change = heading_options[best_heading_change_idx]
    else:
        best_heading_change = 0.0

    return (best_velocity, best_heading_change)

#------------------------------------------------------------------------------------------
# ----------------- Simulation Setup & Loop (Revised Belief Handling) --------------------
#------------------------------------------------------------------------------------------
# --- Rendezvous Config Generation ---
def identify_rendezvous_configs(num_goals, num_agents):
    """Generate configurations where all agents target the same goal."""
    # (Unchanged from previous corrected version)
    if num_goals <= 0 or num_agents <= 0: return []
    configs = []
    for i in range(num_goals):
        configs.append(tuple([i] * num_agents)) # Config represents target goal index
    return configs

# --- Parse Args (Uses Random Beliefs) ---
def parse_args_by_agent(sim_params):
    """Setup agent parameters including random initial 1D beliefs."""
    # (Uses random initial beliefs and gets configs from sim_params)
    agent_vars_list = []
    num_agents = len(sim_params['initial_positions'])
    num_goals = len(sim_params['goals'])
    rendezvous_configs = sim_params.get('rendezvous_configs', []) # Get from global params
    num_configs = len(rendezvous_configs) # Should equal num_goals for rendezvous

    for agent_id in range(num_agents):
        if num_configs > 0:
            initial_beliefs = np.ones(num_goals) / num_goals # Random 1D belief
        else: initial_beliefs = np.array([])

        agent_dict = {
            'agent_id': agent_id, 'num_agents': num_agents,
            'agent_type': sim_params['agent_types'][agent_id],
            'agent_types': sim_params['agent_types'],
            'goals': np.array(sim_params['goals']),
            'velocity_options': sim_params['velocity_options'],
            'heading_options': sim_params['heading_options'],
            'num_actions': len(sim_params['velocity_options']) * len(sim_params['heading_options']),
            'max_distance_measure': sim_params['convergence_distance'],
            'observation_error_std': sim_params['observation_error_std'],
            'beliefs': initial_beliefs, # Assign random 1D belief
            'rendezvous_configs': rendezvous_configs, # Pass configs (used by EFE)
            'use_ep': sim_params['use_epistemic_reasoning'],
            'dt': sim_params['dt'],
            'env_size': sim_params['env_size'],
            'agent_positions': np.array(sim_params['initial_positions'], dtype=float) # Set initial pos here
        }
        agent_vars_list.append(agent_dict)
    return agent_vars_list

# --- Convergence Check ---
def check_rendezvous_convergence(positions, goals, convergence_distance):
    """Check if all agents converged to the same goal."""
    # (Unchanged from previous corrected version)
    num_agents = positions.shape[0]; num_goals = goals.shape[0]
    if num_agents == 0 or num_goals == 0: return False, -1
    distances_to_goals = compute_distances(positions[:, :2], goals[:, :2])
    if distances_to_goals.size == 0 or np.any(np.isnan(distances_to_goals)): return False, -1
    if distances_to_goals.shape != (num_agents, num_goals): return False, -1

    closest_goal_indices = np.argmin(distances_to_goals, axis=1)
    min_distances = np.min(distances_to_goals, axis=1)
    all_close_enough = (min_distances < convergence_distance).all()
    if not all_close_enough: return False, -1
    target_goal = closest_goal_indices[0]
    all_same_goal = (closest_goal_indices == target_goal).all()
    return all_same_goal, target_goal if all_same_goal else -1


# --- Main Simulation Loop ---
def run_simulation(sim_params):
    """Run the simulation using Active Inference."""
    # (Revised logging and belief update logic)
    results = {'params': sim_params, 'log': []}
    agent_vars_list = parse_args_by_agent(sim_params)
    num_agents = sim_params['num_agents']
    max_iterations = sim_params['max_iterations']
    convergence_distance = sim_params['convergence_distance']
    goals = np.array(sim_params['goals'])
    dt = sim_params['dt']
    if num_agents == 0: return results

    current_true_positions = np.array(sim_params['initial_positions'], dtype=float)
    converged = False
    target_goal_idx = -1
    iteration = 0

    start_sim_time = time.time()
    for iteration in range(max_iterations):
        # Log state *before* decision and update
        iteration_log = {
            'iteration': iteration,
            'positions': np.copy(current_true_positions),
             # Log beliefs BEFORE this iteration's decision/update
            'beliefs': [np.copy(av['beliefs']) if av['beliefs'] is not None else np.array([]) for av in agent_vars_list],
            'actions': [None] * num_agents, # Placeholder
            'efe_scores': [np.nan] * num_agents, # Placeholder
        }

        start_iter_time = time.time()
        next_beliefs_all_agents = [None] * num_agents
        decisions = [None] * num_agents

        # --- Decision Making ---
        for agent_id in range(num_agents):
            agent_vars_list[agent_id]['agent_positions'] = np.copy(current_true_positions)
            # Decision function returns action AND the belief state *after* that action
            if sim_params['algorithm'] == 'greedy':
                action = greedy_decision(agent_id, current_true_positions, goals, agent_vars_list[agent_id])
                # Beliefs are not updated in greedy mode
                next_belief = agent_vars_list[agent_id]['beliefs'] # Keep old belief
                efe_score = np.nan # No EFE for greedy
            elif sim_params['algorithm'] == 'active_inference':
                action, next_belief, efe_score = make_decision_active_inference(agent_vars_list[agent_id])
            decisions[agent_id] = action
            next_beliefs_all_agents[agent_id] = next_belief # Store the predicted posterior
            iteration_log['actions'][agent_id] = action
            iteration_log['efe_scores'][agent_id] = efe_score


        # --- State Update ---
        if None in decisions:
             print(f"\nError: Decision failed at iteration {iteration}. Aborting.")
             results['log'].append(iteration_log)
             break
        for agent_id, decision in enumerate(decisions):
            velocity, heading_change = decision
            current_true_positions[agent_id] = predict_agent_position(
                current_true_positions[agent_id], velocity, heading_change, dt
            )

        # --- Belief Update for *NEXT* iteration's prior ---
        # The belief state IS the predicted posterior from the decision step
        for agent_id in range(num_agents):
             num_goals = len(agent_vars_list[agent_id].get('goals', []))
             # Validate the shape of the belief before assigning
             if next_beliefs_all_agents[agent_id] is not None and next_beliefs_all_agents[agent_id].shape == (num_goals,):
                 agent_vars_list[agent_id]['beliefs'] = next_beliefs_all_agents[agent_id]
             else:
                  # Keep old belief if the update failed
                  print(f"Warning: Agent {agent_id} keeping old belief at iter {iteration} due to invalid next_belief.")
                  pass # Belief remains unchanged


        # --- Logging & Convergence Check ---
        results['log'].append(iteration_log) # Log the state *before* this iteration happened
        converged, target_goal_idx = check_rendezvous_convergence(current_true_positions, goals, convergence_distance)

        end_iter_time = time.time()
        iter_time = end_iter_time - start_iter_time
        # Safely access belief for printing
        belief_agent0 = agent_vars_list[0]['beliefs'] if num_agents > 0 and agent_vars_list[0]['beliefs'] is not None and agent_vars_list[0]['beliefs'].size > 0 else None
        belief_agent0_str = f"B0:{np.argmax(belief_agent0)}" if belief_agent0 is not None else "B0:N/A"
        print(f"\rIter {iteration+1}/{max_iterations}|T:{iter_time:.3f}s|Conv:{converged}(G:{target_goal_idx if converged else 'N'})|{belief_agent0_str}", end='')


        if converged:
            print(f"\nConvergence reached at iteration {iteration + 1} to goal {target_goal_idx}!")
            # Add final state log entry
            final_log = {
                'iteration': iteration + 1,
                'positions': np.copy(current_true_positions),
                'beliefs': [np.copy(av['beliefs']) if av['beliefs'] is not None else np.array([]) for av in agent_vars_list],
                'actions': [], 'efe_scores': []
            }
            results['log'].append(final_log)
            break

    final_iteration = iteration + 1
    if not converged:
        print(f"\nSimulation finished after {final_iteration} iterations without convergence.")

    end_sim_time = time.time()
    results['total_time'] = end_sim_time - start_sim_time
    results['converged'] = converged
    results['final_iteration'] = final_iteration
    results['final_positions'] = current_true_positions
    results['final_target_goal'] = target_goal_idx

    return results


#------------------------------------------------------------------------------------------
# ----------------- Plotting Class (Adapted for 1D Beliefs) ------------------------------
#------------------------------------------------------------------------------------------
class SimulationVisualizer:
    """Handles plotting trajectory, beliefs (1D), and EFE."""
    # (Largely unchanged from previous version, ensured compatibility with 1D beliefs)

    def __init__(self, results, save_dir="plots"):
        self.results = results
        self.sim_params = results.get('params', {}) # Use .get for safety
        self.log = results.get('log', [])
        self.num_agents = self.sim_params.get('num_agents', 0)
        self.goals = np.array(self.sim_params.get('goals', []))
        self.num_goals = len(self.goals) # Number of goals = number of belief states for rendezvous
        self.agent_types = self.sim_params.get('agent_types', [])
        self.env_size = self.sim_params.get('env_size', 10) # Default size
        self.padding = self.env_size * 0.1
        # No need for rendezvous_configs here, belief index maps directly to goal index
        self.save_dir = save_dir

        if self.num_agents > 0:
             try:
                cmap = plt.colormaps.get_cmap('tab10') # Use updated API
                self.colors = cmap(np.linspace(0, 1, self.num_agents))
             except ValueError:
                 self.colors = [plt.cm.viridis(i/self.num_agents) for i in range(self.num_agents)]
        else: self.colors = []
        self.markers = ['o', '^', 's', 'P', '*', 'X', 'D', 'v']

        if save_dir and not os.path.exists(self.save_dir):
            try: os.makedirs(self.save_dir)
            except OSError as e: print(f"Error creating save directory {self.save_dir}: {e}"); self.save_dir = None

    def plot_all(self, show=True, save=False):
        """Generate all plots."""
        if not self.log: print("No log data to plot."); return
        can_save = save and self.save_dir is not None
        self._plot_trajectory(save=can_save)
        self._plot_beliefs(save=can_save)
        self._plot_efe(save=can_save)
        if show: plt.show()
        plt.close('all')

    def _plot_trajectory(self, save=False):
        """Plots the agent trajectories."""
        # (Code mostly unchanged from previous corrected version)
        if self.num_agents == 0: return
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-self.padding, self.env_size + self.padding)
        ax.set_ylim(-self.padding, self.env_size + self.padding)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X Position (units)")
        ax.set_ylabel("Y Position (units)")
        title = f"Agent Trajectories (N={self.num_agents}, Goals={self.num_goals}"
        if 'use_epistemic_reasoning' in self.sim_params: title += f", Epistemic={self.sim_params['use_epistemic_reasoning']})"
        else: title += ")"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)

        if self.goals.size > 0:
            ax.plot(self.goals[:, 0], self.goals[:, 1], 'kx', markersize=10, markeredgewidth=1.5, label='Goals', zorder=1)

        for i in range(self.num_agents):
            positions_list = [log_entry['positions'][i] for log_entry in self.log if len(log_entry.get('positions', [])) > i]
            if not positions_list: continue
            positions = np.array(positions_list)
            if positions.shape[1] < 2: continue

            marker = self.markers[i % len(self.markers)]
            agent_type_str = f" ({self.agent_types[i]})" if i < len(self.agent_types) else ""
            label = f'Agent {i}{agent_type_str}'
            color = self.colors[i % len(self.colors)]

            ax.plot(positions[:, 0], positions[:, 1], '-', color=color, linewidth=1.5, alpha=0.7, label=label, zorder=2)
            ax.plot(positions[0, 0], positions[0, 1], marker=marker, color=color, markersize=8, markeredgecolor='k', zorder=3)
            ax.plot(positions[-1, 0], positions[-1, 1], marker=marker, color=color, markersize=10, markerfacecolor='none', markeredgecolor=color, markeredgewidth=1.5, zorder=3)

        ax.legend(fontsize=9, loc='best')
        fig.tight_layout()
        if save:
            ep_str = f"_EP{self.sim_params['use_epistemic_reasoning']}" if 'use_epistemic_reasoning' in self.sim_params else ""
            filename = os.path.join(self.save_dir, f"trajectory_N{self.num_agents}_G{self.num_goals}{ep_str}.png")
            try: fig.savefig(filename, dpi=300); print(f"Trajectory plot saved to {filename}")
            except Exception as e: print(f"Error saving trajectory plot: {e}")
        # plt.close(fig) # Closed in plot_all

    def _plot_beliefs(self, save=False):
        """Plots the 1D belief evolution for each agent."""
        if self.num_agents == 0 or self.num_goals == 0: return

        fig, ax = plt.subplots(figsize=(8, 5))
        iterations = list(range(len(self.log)))
        linestyles = ['-', '--', ':', '-.']
        legend_elements = []
        agent_plotted = [False] * self.num_agents
        goal_plotted = [False] * self.num_goals

        for agent_id in range(self.num_agents):
            # Extract 1D beliefs
            beliefs_list = [log_entry['beliefs'][agent_id] for log_entry in self.log if len(log_entry.get('beliefs', [])) > agent_id and isinstance(log_entry['beliefs'][agent_id], np.ndarray) and log_entry['beliefs'][agent_id].ndim == 1 and log_entry['beliefs'][agent_id].size == self.num_goals]
            if not beliefs_list: continue
            beliefs_over_time = np.array(beliefs_list)
            if beliefs_over_time.shape[1] != self.num_goals: continue # Final shape check

            color = self.colors[agent_id % len(self.colors)]
            if not agent_plotted[agent_id]:
                 legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'Agent {agent_id}'))
                 agent_plotted[agent_id] = True

            # Plot belief probability for each goal
            for goal_idx in range(self.num_goals):
                 linestyle = linestyles[goal_idx % len(linestyles)]
                 ax.plot(iterations[:len(beliefs_over_time)],
                         beliefs_over_time[:, goal_idx], # Index is the goal index
                         color=color,
                         linestyle=linestyle,
                         linewidth=1.5,
                         alpha=0.8)
                 if not goal_plotted[goal_idx]:
                     legend_elements.append(Line2D([0], [0], color='gray', linestyle=linestyle, lw=1.5, label=f'Bel(Goal {goal_idx})'))
                     goal_plotted[goal_idx] = True

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Belief Probability P(Goal | Obs)")
        title = f"Agent Belief Evolution"
        if 'use_epistemic_reasoning' in self.sim_params: title += f" (Epistemic={self.sim_params['use_epistemic_reasoning']})"
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(handles=[h for h in legend_elements if isinstance(h, Line2D)],
                  labels=[h.get_label() for h in legend_elements if isinstance(h, Line2D)],
                  fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
        try: fig.tight_layout(rect=[0, 0, 0.80, 1])
        except ValueError: fig.tight_layout()

        if save:
             ep_str = f"_EP{self.sim_params['use_epistemic_reasoning']}" if 'use_epistemic_reasoning' in self.sim_params else ""
             filename = os.path.join(self.save_dir, f"beliefs_N{self.num_agents}_G{self.num_goals}{ep_str}.png")
             try: fig.savefig(filename, dpi=300); print(f"Belief plot saved to {filename}")
             except Exception as e: print(f"Error saving belief plot: {e}")
        # plt.close(fig) # Closed in plot_all

    def _plot_efe(self, save=False):
        """Plots the chosen EFE score for each agent."""
        # (Code mostly unchanged from previous corrected version)
        if self.num_agents == 0: return
        fig, ax = plt.subplots(figsize=(8, 5))
        iterations = [log_entry['iteration'] for log_entry in self.log] # Use full range

        plot_successful = False
        for agent_id in range(self.num_agents):
             # Extract EFE scores safely, using NaN from log if present
             efe_scores = [log_entry['efe_scores'][agent_id] for log_entry in self.log if len(log_entry.get('efe_scores', [])) > agent_id]
             valid_indices = [it for it, score in enumerate(efe_scores) if np.isfinite(score)]
             valid_efe_scores = [score for score in efe_scores if np.isfinite(score)]

             if not valid_efe_scores: continue # Skip if no valid EFE data

             color = self.colors[agent_id % len(self.colors)]
             agent_type_str = f" ({self.agent_types[agent_id]})" if agent_id < len(self.agent_types) else ""
             label = f'Agent {agent_id}{agent_type_str}'
             ax.plot(valid_indices, valid_efe_scores, '-', color=color, linewidth=1.5, alpha=0.8, label=label)
             plot_successful = True

        if not plot_successful: print("Warning: No valid EFE data to plot."); plt.close(fig); return

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Chosen Action EFE")
        title = f"Expected Free Energy (EFE) of Chosen Actions"
        if 'use_epistemic_reasoning' in self.sim_params: title += f" (Epistemic={self.sim_params['use_epistemic_reasoning']})"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
        fig.tight_layout()
        if save:
            ep_str = f"_EP{self.sim_params['use_epistemic_reasoning']}" if 'use_epistemic_reasoning' in self.sim_params else ""
            filename = os.path.join(self.save_dir, f"efe_N{self.num_agents}_G{self.num_goals}{ep_str}.png")
            try: fig.savefig(filename, dpi=300); print(f"EFE plot saved to {filename}")
            except Exception as e: print(f"Error saving EFE plot: {e}")
        # plt.close(fig) # Closed in plot_all


#------------------------------------------------------------------------------------------
# ----------------- Main Execution Block (Corrected) -------------------------------------
#------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Simulation Parameters ---
    ENV_SIZE = 20.0
    NUM_AGENTS = 3
    NUM_GOALS = 4
    MIN_GOAL_DISTANCE = ENV_SIZE / 4
    CONVERGENCE_DISTANCE = 1.5
    MAX_ITERATIONS = 80
    DT = 0.2

    AGENT_TYPES = ['A', 'A', 'A']
    assert len(AGENT_TYPES) == NUM_AGENTS, "Length of AGENT_TYPES must match NUM_AGENTS"

    OBSERVATION_ERROR_STD = 2

    VELOCITY_OPTIONS = np.array([0.5, 1.0, 1.5])
    HEADING_OPTIONS = np.array([-np.pi/8, 0, np.pi/8])

    USE_EPISTEMIC_REASONING = False #True # Set True for higher-order, False for first-order

    # --- Initialization ---
    goals = generate_spread_out_goals(NUM_GOALS, ENV_SIZE, MIN_GOAL_DISTANCE)
    if len(goals) < NUM_GOALS: exit()
    initial_positions = []
    min_start_dist_goal = ENV_SIZE / 5
    for i in range(NUM_AGENTS):
        attempts = 0
        while attempts < 100:
            pos = np.random.rand(2) * ENV_SIZE; heading = np.random.rand() * 2 * np.pi - np.pi
            if all(np.linalg.norm(pos - g[:2]) >= min_start_dist_goal for g in goals) and \
               all(np.linalg.norm(pos - p[:2]) >= 1.0 for p in initial_positions):
                 initial_positions.append(np.concatenate([pos, [heading]])); break
            attempts += 1
        if len(initial_positions) <= i: print(f"Error: Could not place agent {i}. Exiting."); exit()
    initial_positions = np.array(initial_positions)

    # --- Generate rendezvous_configs HERE ---
    rendezvous_configs = identify_rendezvous_configs(NUM_GOALS, NUM_AGENTS)

    # --- Pack parameters ---
    sim_params = {
        'env_size': ENV_SIZE, 'num_agents': NUM_AGENTS, 'goals': goals.tolist(),
        'algorithm': 'active_inference', # or 'greedy'
        'initial_positions': initial_positions.tolist(), 'agent_types': AGENT_TYPES,
        'velocity_options': VELOCITY_OPTIONS.tolist(), 'heading_options': HEADING_OPTIONS.tolist(),
        'observation_error_std': OBSERVATION_ERROR_STD,
        'use_epistemic_reasoning': USE_EPISTEMIC_REASONING,
        'max_iterations': MAX_ITERATIONS, 'convergence_distance': CONVERGENCE_DISTANCE, 'dt': DT,
        'rendezvous_configs': rendezvous_configs, # Add configs here
    }

    # --- Run Simulation ---
    print("Starting simulation...")
    results = run_simulation(sim_params)
    print(f"\nSimulation finished in {results['total_time']:.2f} seconds.")
    print(f"Converged: {results['converged']}")
    if results['converged']:
        target_goal_idx = results['final_target_goal']
        if 0 <= target_goal_idx < len(goals): print(f"Converged to goal index: {target_goal_idx} at position {goals[target_goal_idx]}")
        else: print(f"Converged to invalid goal index: {target_goal_idx}")

    # --- Plotting ---
    if results.get('log'):
        print("\nGenerating plots...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        ep_str = "EPTrue" if USE_EPISTEMIC_REASONING else "EPFalse"
        save_directory = f"plots_{timestamp}_N{NUM_AGENTS}_G{NUM_GOALS}_{ep_str}"
        visualizer = SimulationVisualizer(results, save_dir=save_directory)
        visualizer.plot_all(show=True, save=True)
    else: print("No simulation log data to plot.")