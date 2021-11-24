use std::f64::consts::PI;

use rand::{seq::SliceRandom, SeedableRng};
use rand_distr::{Bernoulli, Distribution, Normal};
use rand_pcg::Pcg64Mcg;

use hammer_and_sample::{sample, Model};

#[test]
fn hierarchical() {
    const GROUPS: usize = 10;
    const OBSERVATIONS: usize = 1000;
    const WALKERS: usize = 1000;
    const ITERATIONS: usize = 10000;
    const BURN_IN: usize = 5000;

    struct Hierarchical {
        data: Vec<(bool, usize)>,
    }

    impl Model for Hierarchical {
        type Params = [f64; 2 + GROUPS];

        fn log_prob(&self, state: &Self::Params) -> f64 {
            let logit_theta = state[0];
            let sigma = state[1];

            // prior on logit_theta
            let mut log_prob = log_normal(logit_theta, -2., 2.);

            // prior on sigma (half normal)
            if sigma < 0. {
                return f64::NEG_INFINITY;
            }

            log_prob += log_normal(sigma, 0., 2.) + f64::ln(2.);

            let mut ln_group_theta = [0.; GROUPS];
            let mut ln_1_group_theta = [0.; GROUPS];

            for group in 0..GROUPS {
                let group_alpha = state[2 + group];
                let group_theta = expit(logit_theta + group_alpha);

                ln_group_theta[group] = group_theta.ln();
                ln_1_group_theta[group] = (1. - group_theta).ln();

                // likelihood of group_alpha given sigma
                log_prob += log_normal(group_alpha, 0., sigma);
            }

            // likelihood of data given group_theta
            log_prob += self
                .data
                .iter()
                .map(|(data, group)| {
                    if *data {
                        ln_group_theta[*group]
                    } else {
                        ln_1_group_theta[*group]
                    }
                })
                .sum::<f64>();

            log_prob
        }
    }

    let true_logit_theta = -3.;
    let true_sigma = 1.;
    let mut true_alpha = [0.; GROUPS];

    let mut rng = Pcg64Mcg::seed_from_u64(0);

    let mut data = Vec::new();

    let true_alpha_dist = Normal::<f64>::new(0., true_sigma).unwrap();

    for group in 0..GROUPS {
        let true_group_alpha = true_alpha_dist.sample(&mut rng);
        let true_group_theta = expit(true_logit_theta + true_group_alpha);

        true_alpha[group] = true_group_alpha;

        let dist = Bernoulli::new(true_group_theta).unwrap();

        for _ in 0..OBSERVATIONS {
            data.push((dist.sample(&mut rng), group));
        }
    }

    data.shuffle(&mut rng);

    let model = Hierarchical { data };

    let prior_logit_theta = Normal::<f64>::new(-2., 2.).unwrap();
    let prior_sigma = Normal::<f64>::new(0., 2.).unwrap();

    let walkers = (0..WALKERS).map(|_| {
        let mut rng = Pcg64Mcg::from_rng(&mut rng).unwrap();

        let guess_logit_theta = prior_logit_theta.sample(&mut rng);
        let guess_sigma = prior_sigma.sample(&mut rng).abs();

        let mut guess = [0.; 2 + GROUPS];
        guess[0] = guess_logit_theta;
        guess[1] = guess_sigma;

        let prior_alpha = Normal::new(0., guess_sigma).unwrap();

        for group in 0..GROUPS {
            let guess_group_alpha = prior_alpha.sample(&mut rng);

            guess[2 + group] = guess_group_alpha;
        }

        (guess, rng)
    });

    let (chain, accepted) = sample(&model, walkers, ITERATIONS);

    let converged_chain = &chain[WALKERS * BURN_IN..];

    let estimated_logit_theta =
        converged_chain.iter().map(|params| params[0]).sum::<f64>() / converged_chain.len() as f64;

    assert!((true_logit_theta - estimated_logit_theta).abs() < 0.1);

    let estimated_sigma =
        converged_chain.iter().map(|params| params[1]).sum::<f64>() / converged_chain.len() as f64;

    assert!((true_sigma - estimated_sigma).abs() < 0.1);

    for group in 0..GROUPS {
        let estimated_group_alpha = converged_chain
            .iter()
            .map(|params| params[2 + group])
            .sum::<f64>()
            / converged_chain.len() as f64;

        assert!((true_alpha[group] - estimated_group_alpha).abs() < 1.);
    }

    let acceptance_rate = accepted as f64 / chain.len() as f64;

    assert!(acceptance_rate > 0.001);
}

fn expit(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn log_normal(x: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * ((x - mu) / sigma).powi(2) - f64::ln(f64::sqrt(2. * PI) * sigma)
}
