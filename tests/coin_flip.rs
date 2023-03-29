use rand::{
    distributions::{Bernoulli, Distribution, Standard},
    SeedableRng,
};
use rand_pcg::Pcg64Mcg;

use hammer_and_sample::{auto_corr_time, sample, MinChainLen, Model, Serial};

#[test]
fn coin_flip() {
    struct CoinFlip {
        data: Vec<bool>,
    }

    impl Model for CoinFlip {
        type Params = [f64; 1];

        fn log_prob(&self, state: &Self::Params) -> f64 {
            if state[0] < 0. || state[0] > 1. {
                return f64::NEG_INFINITY;
            }

            let ln_p = state[0].ln();
            let ln_1_p = (1. - state[0]).ln();

            self.data
                .iter()
                .map(|data| if *data { ln_p } else { ln_1_p })
                .sum()
        }
    }

    let true_p = 0.75;

    let mut rng = Pcg64Mcg::seed_from_u64(0);

    let dist = Bernoulli::new(true_p).unwrap();

    let model = CoinFlip {
        data: (0..1000).map(|_| dist.sample(&mut rng)).collect(),
    };

    let walkers = (0..100).map(|_| {
        let mut rng = Pcg64Mcg::from_rng(&mut rng).unwrap();

        let guess_p = Distribution::<f64>::sample(&Standard, &mut rng);

        ([guess_p], rng)
    });

    let (chain, accepted) = sample(&model, walkers, MinChainLen(100_000), Serial);

    let converged_chain = &chain[10_000..];

    let estimated_p =
        converged_chain.iter().map(|params| params[0]).sum::<f64>() / converged_chain.len() as f64;

    assert!((true_p - estimated_p).abs() < 0.01);

    let acceptance_rate = accepted as f64 / chain.len() as f64;

    assert!(acceptance_rate > 0.7 && acceptance_rate < 0.8);

    let auto_corr_time_p =
        auto_corr_time(converged_chain.iter().map(|params| params[0]), None, None).unwrap();

    assert!(converged_chain.len() as f64 / auto_corr_time_p > 10_000.);
}
