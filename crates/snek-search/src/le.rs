//! Logit-Equilibrium solver via Stochastic Fictitious Play (SFP).
//!
//! Given a finite normal-form game (each agent has a small set of candidate
//! actions and a per-agent payoff for every joint action), this iterates smooth
//! best responses to a fixed point. Run at a high temperature it approximates a
//! Nash equilibrium ("assume perfect play"); the smoothing keeps the iteration
//! stable and well-defined for general-sum, multi-agent games where exact Nash
//! is messy or non-unique.

use snek_core::MAX_SNAKES;

/// Result of solving one node's game.
pub struct LeSolution {
    /// Per-agent equilibrium expected value (length = number of agents).
    pub values: [f32; MAX_SNAKES],
    /// Per-agent mixed strategy over that agent's candidate actions.
    pub policies: Vec<Vec<f32>>,
}

/// Solve the logit equilibrium of a normal-form game.
///
/// * `cand_lens[i]` — number of candidate actions for agent `i`.
/// * `payoffs[joint][i]` — agent `i`'s payoff for the joint action `joint`,
///   where joint actions are enumerated row-major with agent 0 most significant
///   (`joint = sum_i a_i * stride_i`, `stride_i = prod_{k>i} cand_lens[k]`).
/// * `tau` — per-agent inverse temperature (higher ⇒ sharper ⇒ closer to best
///   response). Length must equal `cand_lens.len()`. Heterogeneous temperatures
///   give a quantal-response / SBRLE-style equilibrium: a rational agent at a
///   high `tau` best-responds to weaker agents at lower `tau`.
/// * `iters` — number of SFP iterations.
pub fn solve(
    cand_lens: &[usize],
    payoffs: &[[f32; MAX_SNAKES]],
    tau: &[f32],
    iters: usize,
) -> LeSolution {
    debug_assert_eq!(tau.len(), cand_lens.len());
    let n = cand_lens.len();
    let total: usize = cand_lens.iter().product();
    debug_assert_eq!(payoffs.len(), total);

    // Strides for decoding joint indices.
    let mut stride = vec![1usize; n];
    for i in (0..n).rev() {
        stride[i] = if i + 1 < n {
            stride[i + 1] * cand_lens[i + 1]
        } else {
            1
        };
    }

    // Uniform initial strategies.
    let mut pi: Vec<Vec<f32>> = cand_lens.iter().map(|&l| vec![1.0 / l as f32; l]).collect();

    // Scratch for the per-agent action-value vectors q_i.
    let mut q: Vec<Vec<f32>> = cand_lens.iter().map(|&l| vec![0.0; l]).collect();

    let compute_q = |pi: &[Vec<f32>], q: &mut [Vec<f32>]| {
        for qi in q.iter_mut() {
            qi.iter_mut().for_each(|x| *x = 0.0);
        }
        for (joint, payoff) in payoffs.iter().enumerate().take(total) {
            for i in 0..n {
                let ai = (joint / stride[i]) % cand_lens[i];
                // Weight = product of the other agents' probabilities.
                let mut w = 1.0f32;
                for k in 0..n {
                    if k != i {
                        let ak = (joint / stride[k]) % cand_lens[k];
                        w *= pi[k][ak];
                    }
                }
                q[i][ai] += w * payoff[i];
            }
        }
    };

    for t in 0..iters {
        compute_q(&pi, &mut q);
        let alpha = 1.0 / (t as f32 + 2.0);
        for i in 0..n {
            // Smooth best response: softmax(tau * q_i), numerically stabilized.
            let qi = &q[i];
            let max = qi.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut sbr = vec![0.0f32; qi.len()];
            for (a, &qa) in qi.iter().enumerate() {
                let e = ((qa - max) * tau[i]).exp();
                sbr[a] = e;
                denom += e;
            }
            for a in 0..qi.len() {
                sbr[a] /= denom;
                // Move pi toward the smooth best response.
                pi[i][a] += alpha * (sbr[a] - pi[i][a]);
            }
        }
    }

    // Final values under the converged strategies.
    compute_q(&pi, &mut q);
    let mut values = [0.0f32; MAX_SNAKES];
    for i in 0..n {
        let mut v = 0.0f32;
        for a in 0..cand_lens[i] {
            v += pi[i][a] * q[i][a];
        }
        values[i] = v;
    }

    LeSolution {
        values,
        policies: pi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payoff_vec(entries: &[[f32; 2]]) -> Vec<[f32; MAX_SNAKES]> {
        entries
            .iter()
            .map(|e| {
                let mut p = [0.0; MAX_SNAKES];
                p[0] = e[0];
                p[1] = e[1];
                p
            })
            .collect()
    }

    #[test]
    fn matching_pennies_is_uniform() {
        // Zero-sum, no pure equilibrium: unique mixed equilibrium is (0.5, 0.5).
        // joint order (a0,a1): (0,0),(0,1),(1,0),(1,1)
        // U0: match -> +1, mismatch -> -1; U1 = -U0.
        let payoffs = payoff_vec(&[[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]);
        let sol = solve(&[2, 2], &payoffs, &[6.0, 6.0], 500);
        for i in 0..2 {
            assert!(
                (sol.policies[i][0] - 0.5).abs() < 0.05,
                "agent {i} ~ uniform"
            );
        }
        // Game value is ~0 for both.
        assert!(sol.values[0].abs() < 0.1 && sol.values[1].abs() < 0.1);
    }

    #[test]
    fn dominant_action_is_selected() {
        // Agent 0 action 0 strictly dominates; agent 1 action 1 strictly dominates.
        // U0 high when a0=0; U1 high when a1=1.
        let payoffs = payoff_vec(&[
            [1.0, 0.0], // (0,0)
            [1.0, 1.0], // (0,1)
            [0.0, 0.0], // (1,0)
            [0.0, 1.0], // (1,1)
        ]);
        let sol = solve(&[2, 2], &payoffs, &[8.0, 8.0], 500);
        assert!(sol.policies[0][0] > 0.95, "agent 0 picks dominant action 0");
        assert!(sol.policies[1][1] > 0.95, "agent 1 picks dominant action 1");
    }

    #[test]
    fn single_action_agent_is_trivial() {
        // Agent 1 has only one action; agent 0 prefers action 1 here.
        let mut payoffs = vec![[0.0; MAX_SNAKES]; 2];
        payoffs[0][0] = -1.0; // a0=0
        payoffs[1][0] = 1.0; // a0=1
        let sol = solve(&[2, 1], &payoffs, &[8.0, 8.0], 300);
        assert!(sol.policies[1][0] > 0.999);
        assert!(
            sol.policies[0][1] > 0.95,
            "agent 0 prefers the better action"
        );
    }

    #[test]
    fn high_tau_sharpens_toward_best_response() {
        let payoffs = payoff_vec(&[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
        let low = solve(&[2, 2], &payoffs, &[1.0, 1.0], 500);
        let high = solve(&[2, 2], &payoffs, &[12.0, 12.0], 500);
        assert!(
            high.policies[0][0] > low.policies[0][0],
            "higher tau concentrates more on the better action"
        );
    }

    #[test]
    fn heterogeneous_tau_lets_rational_agent_exploit_weak_one() {
        // Coordination-ish game: agent 0 prefers action 0; agent 1's payoff is
        // flat (indifferent), so at low tau it plays ~uniform. A rational agent 0
        // (high tau) should then concentrate harder on its best action than it
        // would against a rational agent 1.
        let payoffs = payoff_vec(&[[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
        // Agent 1 weak (tau 0.5), agent 0 rational (tau 12).
        let exploit = solve(&[2, 2], &payoffs, &[12.0, 0.5], 500);
        // Agent 1 near-uniform because it's near-indifferent and low-tau.
        assert!(
            (exploit.policies[1][0] - 0.5).abs() < 0.15,
            "weak agent ~uniform"
        );
        // Agent 0 strongly prefers action 0 (its better action given a1 uniform).
        assert!(exploit.policies[0][0] > 0.9, "rational agent exploits");
    }
}
