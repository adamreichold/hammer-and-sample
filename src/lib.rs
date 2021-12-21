#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Simplistic MCMC ensemble sampler based on [emcee](https://emcee.readthedocs.io/), the MCMC hammer
//!
//! ```
//! use hammer_and_sample::{sample, Model, Serial};
//! use rand::{Rng, SeedableRng};
//! use rand_pcg::Pcg64;
//!
//! fn estimate_bias(coin_flips: &[bool]) -> f64 {
//!     struct CoinFlips<'a>(&'a [bool]);
//!
//!     impl Model for CoinFlips<'_> {
//!         type Params = [f64; 1];
//!
//!         // likelihood of Bernoulli distribution and uninformative prior
//!         fn log_prob(&self, &[p]: &Self::Params) -> f64 {
//!             if p < 0. || p > 1. {
//!                 return f64::NEG_INFINITY;
//!             }
//!
//!             let ln_p = p.ln();
//!             let ln_1_p = (1. - p).ln();
//!
//!             self.0
//!                 .iter()
//!                 .map(|coin_flip| if *coin_flip { ln_p } else { ln_1_p })
//!                 .sum()
//!         }
//!     }
//!
//!     let model = CoinFlips(coin_flips);
//!
//!     let walkers = (0..10).map(|seed| {
//!         let mut rng = Pcg64::seed_from_u64(seed);
//!
//!         let p = rng.gen_range(0.0..=1.0);
//!
//!         ([p], rng)
//!     });
//!
//!     let (chain, _accepted) = sample(&model, walkers, 1000, &Serial);
//!
//!     // 100 iterations of 10 walkers as burn-in
//!     let chain = &chain[10 * 100..];
//!
//!     chain.iter().map(|&[p]| p).sum::<f64>() / chain.len() as f64
//! }
//! ```
use rand::{
    distributions::{Distribution, Standard, Uniform},
    Rng,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelExtend, ParallelIterator};

/// Model parameters defining the state space of the Markov chain.
pub trait Params: Send + Sync + Clone {
    /// Dimension of the state space
    ///
    /// This can depend on the instance for situations where the number of parameters depends on the data, e.g. the number of groups in a hierarchical model.
    fn dimension(&self) -> usize;

    /// Propose a new parameters doing a stretch move based on the parameters `other` and the scale `z`
    #[must_use]
    fn propose(&self, other: &Self, z: f64) -> Self;
}

/// Model parameters stored as an array of length `N` considered as an element of the vector space `R^N`.
impl<const N: usize> Params for [f64; N] {
    fn dimension(&self) -> usize {
        N
    }

    fn propose(&self, other: &Self, z: f64) -> Self {
        let mut new = [0.; N];
        for i in 0..N {
            new[i] = other[i] - z * (other[i] - self[i]);
        }
        new
    }
}

/// Model parameters stored as a vector of length `n` considered as an element of the vector space `R^n`.
impl Params for Vec<f64> {
    fn dimension(&self) -> usize {
        self.len()
    }

    fn propose(&self, other: &Self, z: f64) -> Self {
        self.iter()
            .zip(other)
            .map(|(self_, other)| other - z * (other - self_))
            .collect()
    }
}

/// Models are defined by the type of their parameters and their probability functions
pub trait Model: Send + Sync {
    /// Type used to store the model parameters, e.g. `[f64; N]` or `Vec<f64>`
    type Params: Params;

    /// The logarithm of the probability determined by the model given the parameters stored in `state`, up to an addititive constant
    ///
    /// The sampler will only ever consider differences of these values, i.e. any addititive constant that does _not_ depend on `state` can be omitted when computing this.
    fn log_prob(&self, state: &Self::Params) -> f64;

    /// Scale parameter for stretch moves
    const SCALE: f64 = 2.;
}

/// Runs the sampler for `iterations` of the given `model` using the chosen `execution` strategy
///
/// The `walkers` iterator is used to initialise the ensemble of walkers
/// by defining their initial parameter values and providing appropriately seeded PRNG instances.
///
/// The number of walkers must be non-zero, even and at least twice the number of parameters.
///
/// A vector of samples and the number of accepted moves are returned.
pub fn sample<M, W, R, E>(
    model: &M,
    walkers: W,
    iterations: usize,
    execution: &E,
) -> (Vec<M::Params>, usize)
where
    M: Model,
    W: Iterator<Item = (M::Params, R)>,
    R: Rng + Send + Sync,
    E: Execution,
{
    let mut walkers = walkers
        .map(|(state, rng)| Walker::new(model, state, rng))
        .collect::<Vec<_>>();

    assert!(!walkers.is_empty() && walkers.len() % 2 == 0);
    assert!(walkers.len() >= 2 * walkers[0].state.dimension());

    let mut chain = Vec::with_capacity(walkers.len() * iterations);

    let half = walkers.len() / 2;
    let (lower_half, upper_half) = walkers.split_at_mut(half);

    let random_index = Uniform::new(0, half);

    let update_walker = |walker: &mut Walker<M, R>, other_walkers: &[Walker<M, R>]| {
        let other = &other_walkers[random_index.sample(&mut walker.rng)];

        walker.move_(model, other)
    };

    for _ in 0..iterations {
        execution.extend_chain(&mut chain, lower_half, |walker| {
            update_walker(walker, upper_half)
        });

        execution.extend_chain(&mut chain, upper_half, |walker| {
            update_walker(walker, lower_half)
        });
    }

    let accepted = walkers.iter().map(|walker| walker.accepted).sum();

    (chain, accepted)
}

struct Walker<M, R>
where
    M: Model,
{
    state: M::Params,
    log_prob: f64,
    rng: R,
    accepted: usize,
}

impl<M, R> Walker<M, R>
where
    M: Model,
    R: Rng,
{
    fn new(model: &M, state: M::Params, rng: R) -> Self {
        let log_prob = model.log_prob(&state);

        Self {
            state,
            log_prob,
            rng,
            accepted: 0,
        }
    }

    fn move_(&mut self, model: &M, other: &Self) -> M::Params {
        let z = ((M::SCALE - 1.) * gen_unit(&mut self.rng) + 1.).powi(2) / M::SCALE;

        let mut new_state = self.state.propose(&other.state, z);
        let new_log_prob = model.log_prob(&new_state);

        let log_prob_diff =
            (new_state.dimension() - 1) as f64 * z.ln() + new_log_prob - self.log_prob;

        if log_prob_diff > gen_unit(&mut self.rng).ln() {
            self.state.clone_from(&new_state);
            self.log_prob = new_log_prob;
            self.accepted += 1;
        } else {
            new_state.clone_from(&self.state);
        }

        new_state
    }
}

fn gen_unit<R>(rng: &mut R) -> f64
where
    R: Rng,
{
    Distribution::<f64>::sample(&Standard, rng)
}

/// Execution strategy for `update`ing an ensemble of `walkers` to extend the given `chain`
pub trait Execution {
    /// Must call `update` exactly once for all elements of `walkers` and store the results in `chain`
    fn extend_chain<P, W, U>(&self, chain: &mut Vec<P>, walkers: &mut [W], update: U)
    where
        P: Send + Sync,
        W: Send + Sync,
        U: Fn(&mut W) -> P + Send + Sync;
}

/// Serial execution strategy which updates walkers using a single thread
pub struct Serial;

impl Execution for Serial {
    fn extend_chain<P, W, U>(&self, chain: &mut Vec<P>, walkers: &mut [W], update: U)
    where
        P: Send + Sync,
        W: Send + Sync,
        U: Fn(&mut W) -> P + Send + Sync,
    {
        chain.extend(walkers.iter_mut().map(update));
    }
}

#[cfg(feature = "rayon")]
/// Parallel execution strategy which updates walkers using Rayon's thread pool
pub struct Parallel;

#[cfg(feature = "rayon")]
impl Execution for Parallel {
    fn extend_chain<P, W, U>(&self, chain: &mut Vec<P>, walkers: &mut [W], update: U)
    where
        P: Send + Sync,
        W: Send + Sync,
        U: Fn(&mut W) -> P + Send + Sync,
    {
        chain.par_extend(walkers.par_iter_mut().map(update));
    }
}
