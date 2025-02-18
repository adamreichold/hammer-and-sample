#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Simplistic MCMC ensemble sampler based on [emcee](https://emcee.readthedocs.io/), the MCMC hammer
//!
//! ```
//! use hammer_and_sample::{sample, MinChainLen, Model, Serial, Stretch};
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
//!         let p = rng.random_range(0.0..=1.0);
//!
//!         ([p], rng)
//!     });
//!
//!     let (chain, _accepted) = sample(&model, &Stretch::default(), walkers, MinChainLen(10 * 1000), Serial);
//!
//!     // 100 iterations of 10 walkers as burn-in
//!     let chain = &chain[10 * 100..];
//!
//!     chain.iter().map(|&[p]| p).sum::<f64>() / chain.len() as f64
//! }
//! ```
use std::ops::ControlFlow;
use std::ptr;

use rand::{
    distr::{Distribution, StandardUniform, Uniform},
    Rng,
};
use rand_distr::{
    weighted::{AliasableWeight, WeightedAliasIndex},
    Normal,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelExtend, ParallelIterator};

/// Model parameters defining the state space of the Markov chain
pub trait Params: Send + Sync + Clone {
    /// Dimension of the state space
    ///
    /// This can depend on `self` in situations where the number of parameters depends on the data itself, e.g. the number of groups in a hierarchical model.
    fn dimension(&self) -> usize;

    /// Access the individual parameter values as an iterator
    fn values(&self) -> impl Iterator<Item = &f64>;

    /// Collect new parameters from the given iterator
    fn collect(iter: impl Iterator<Item = f64>) -> Self;
}

/// Model parameters stored as an array of length `N` considered as an element of the vector space `R^N`
impl<const N: usize> Params for [f64; N] {
    fn dimension(&self) -> usize {
        N
    }

    fn values(&self) -> impl Iterator<Item = &f64> {
        self.iter()
    }

    fn collect(iter: impl Iterator<Item = f64>) -> Self {
        let mut new = [0.; N];
        iter.enumerate().for_each(|(idx, value)| new[idx] = value);
        new
    }
}

/// Model parameters stored as a vector of length `n` considered as an element of the vector space `R^n`
impl Params for Vec<f64> {
    fn dimension(&self) -> usize {
        self.len()
    }

    fn values(&self) -> impl Iterator<Item = &f64> {
        self.iter()
    }

    fn collect(iter: impl Iterator<Item = f64>) -> Self {
        iter.collect()
    }
}

/// Model parameters stored as a boxed slice of length `n` considered as an element of the vector space `R^n`
impl Params for Box<[f64]> {
    fn dimension(&self) -> usize {
        self.len()
    }

    fn values(&self) -> impl Iterator<Item = &f64> {
        self.iter()
    }

    fn collect(iter: impl Iterator<Item = f64>) -> Self {
        iter.collect()
    }
}

/// TODO
pub trait Move<M>
where
    M: Model,
{
    /// TODO
    fn propose<'a, O, R>(&self, self_: &'a M::Params, other: O, rng: &mut R) -> (M::Params, f64)
    where
        O: FnMut(&mut R) -> &'a M::Params,
        R: Rng;
}

/// TODO
pub struct Stretch {
    scale: f64,
}

impl Stretch {
    /// TODO
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl Default for Stretch {
    fn default() -> Self {
        Self::new(2.)
    }
}

impl<M> Move<M> for Stretch
where
    M: Model,
{
    fn propose<'a, O, R>(&self, self_: &'a M::Params, mut other: O, rng: &mut R) -> (M::Params, f64)
    where
        O: FnMut(&mut R) -> &'a M::Params,
        R: Rng,
    {
        let other = other(rng);

        let z = ((self.scale - 1.) * gen_unit(rng) + 1.).powi(2) / self.scale;

        let new_state = M::Params::collect(
            self_
                .values()
                .zip(other.values())
                .map(|(self_, other)| other - z * (other - self_)),
        );

        let factor = (new_state.dimension() - 1) as f64 * z.ln();

        (new_state, factor)
    }
}

/// TODO
pub struct DifferentialEvolution {
    gamma: Normal<f64>,
}

impl DifferentialEvolution {
    /// TODO
    pub fn new(gamma_mean: f64, gamma_std_dev: f64) -> Self {
        Self {
            gamma: Normal::new(gamma_mean, gamma_std_dev).unwrap(),
        }
    }
}

impl<M> Move<M> for DifferentialEvolution
where
    M: Model,
{
    fn propose<'a, O, R>(&self, self_: &'a M::Params, mut other: O, rng: &mut R) -> (M::Params, f64)
    where
        O: FnMut(&mut R) -> &'a M::Params,
        R: Rng,
    {
        let first_other = other(rng);
        let mut second_other = other(rng);

        while ptr::eq(first_other, second_other) {
            second_other = other(rng);
        }

        let gamma = self.gamma.sample(rng);

        let new_state = M::Params::collect(
            self_
                .values()
                .zip(first_other.values())
                .zip(second_other.values())
                .map(|((self_, first_other), second_other)| {
                    self_ + gamma * (first_other - second_other)
                }),
        );

        (new_state, 0.)
    }
}

/// TODO
pub struct RandomGaussian {
    dist: Normal<f64>,
}

impl RandomGaussian {
    /// TODO
    pub fn new(scale: f64) -> Self {
        Self {
            dist: Normal::new(0., scale).unwrap(),
        }
    }
}

impl<M> Move<M> for RandomGaussian
where
    M: Model,
{
    fn propose<'a, O, R>(&self, self_: &'a M::Params, _other: O, rng: &mut R) -> (M::Params, f64)
    where
        O: FnMut(&mut R) -> &'a M::Params,
        R: Rng,
    {
        let dir = rng.random_range(0..self_.dimension());

        let new_state = M::Params::collect(self_.values().enumerate().map(|(idx, value)| {
            if idx == dir {
                value + self.dist.sample(rng)
            } else {
                *value
            }
        }));

        (new_state, 0.)
    }
}

/// TODO
pub struct Mixture<W, M>(WeightedAliasIndex<W>, M)
where
    W: AliasableWeight;

macro_rules! impl_mixture {
    ( $( $types:ident @ $weights:ident),+ ) => {
        impl<W, $( $types ),+> From<( $( ( $types, W ) ),+ )> for Mixture<W, ( $( $types ),+ )>
        where
            W: AliasableWeight
        {
            #[allow(non_snake_case)]
            fn from(( $( ( $types, $weights ) ),+ ): ( $( ( $types, W ) ),+ )) -> Self {
                let index = WeightedAliasIndex::new(vec![$( $weights ),+]).unwrap();

                Self(index, ( $( $types ),+ ))
            }
        }

        impl<W, $( $types ),+, M> Move<M> for Mixture<W, ( $( $types ),+ )>
        where
            W: AliasableWeight,
            M: Model,
            $( $types: Move<M> ),+
        {
            #[allow(non_snake_case)]
            fn propose<'a, O, R>(&self, self_: &'a M::Params, other: O, rng: &mut R) -> (M::Params, f64)
            where
                O: FnMut(&mut R) -> &'a M::Params,
                R: Rng,
            {
                let Self(index, ( $( $types ),+ )) = self;

                let chosen_index = index.sample(rng);

                let mut index = 0;

                $(

                #[allow(unused_assignments)]
                if chosen_index == index {
                    return $types.propose(self_, other, rng)
                } else {
                    index += 1;
                }

                )+

                unreachable!()
            }
        }
    };
}

impl_mixture!(A @ a, B @ b);
impl_mixture!(A @ a, B @ b, C @ c);
impl_mixture!(A @ a, B @ b, C @ c, D @ d);
impl_mixture!(A @ a, B @ b, C @ c, D @ d, E @ e);
impl_mixture!(A @ a, B @ b, C @ c, D @ d, E @ e, F @ f);
impl_mixture!(A @ a, B @ b, C @ c, D @ d, E @ e, F @ f, G @ g);
impl_mixture!(A @ a, B @ b, C @ c, D @ d, E @ e, F @ f, G @ g, H @ h);
impl_mixture!(A @ a, B @ b, C @ c, D @ d, E @ e, F @ f, G @ g, H @ h, I @ i);
impl_mixture!(A @ a, B @ b, C @ c, D @ d, E @ e, F @ f, G @ g, H @ h, I @ i, J @ j);

/// Models are defined by the type of their parameters and their probability functions
pub trait Model: Send + Sync {
    /// Type used to store the model parameters, e.g. `[f64; N]` or `Vec<f64>`
    type Params: Params;

    /// The logarithm of the probability determined by the model given the parameters stored in `state`, up to an addititive constant
    ///
    /// The sampler will only ever consider differences of these values, i.e. any addititive constant that does _not_ depend on `state` can be omitted when computing them.
    fn log_prob(&self, state: &Self::Params) -> f64;
}

/// Runs the sampler on the given [`model`][Model] using the chosen [`schedule`][Schedule] and [`execution`][Execution] strategy
///
/// The `walkers` iterator is used to initialise the ensemble of walkers by defining their initial parameter values and providing appropriately seeded PRNG instances.
///
/// The number of walkers must be non-zero, even and at least twice the number of parameters.
///
/// A vector of samples and the number of accepted moves are returned.
pub fn sample<MD, MV, W, R, S, E>(
    model: &MD,
    move_: &MV,
    walkers: W,
    mut schedule: S,
    execution: E,
) -> (Vec<MD::Params>, usize)
where
    MD: Model,
    MV: Move<MD> + Send + Sync,
    W: Iterator<Item = (MD::Params, R)>,
    R: Rng + Send + Sync,
    S: Schedule<MD::Params>,
    E: Execution,
{
    let mut walkers = walkers
        .map(|(state, rng)| Walker::new(model, state, rng))
        .collect::<Vec<_>>();

    assert!(!walkers.is_empty() && walkers.len() % 2 == 0);
    assert!(walkers.len() >= 2 * walkers[0].state.dimension());

    let mut chain =
        Vec::with_capacity(walkers.len() * schedule.iterations(walkers.len()).unwrap_or(0));

    let half = walkers.len() / 2;
    let (lower_half, upper_half) = walkers.split_at_mut(half);

    let random_index = Uniform::new(0, half).unwrap();

    let update_walker = move |walker: &mut Walker<MD, R>, other_walkers: &[Walker<MD, R>]| {
        walker.move_(model, move_, |rng| &other_walkers[random_index.sample(rng)])
    };

    while schedule.next_step(&chain).is_continue() {
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

struct Walker<MD, R>
where
    MD: Model,
{
    state: MD::Params,
    log_prob: f64,
    rng: R,
    accepted: usize,
}

impl<MD, R> Walker<MD, R>
where
    MD: Model,
    R: Rng,
{
    fn new(model: &MD, state: MD::Params, rng: R) -> Self {
        let log_prob = model.log_prob(&state);

        Self {
            state,
            log_prob,
            rng,
            accepted: 0,
        }
    }

    fn move_<'a, MV, O>(&'a mut self, model: &MD, move_: &MV, mut other: O) -> MD::Params
    where
        MV: Move<MD>,
        O: FnMut(&mut R) -> &'a Self,
    {
        let (mut new_state, factor) =
            move_.propose(&self.state, |rng| &other(rng).state, &mut self.rng);

        let new_log_prob = model.log_prob(&new_state);

        let log_prob_diff = factor + new_log_prob - self.log_prob;

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
    StandardUniform.sample(rng)
}

/// Estimate the integrated auto-correlation time
///
/// Returns `None` if the chain length is considered insufficient for a reliable estimate.
///
/// `min_win_size` defines the factor between the estimate and the window size up to which the auto-correlation is computed. (Default value: 5)
///
/// `min_chain_len` defines the factor between the estimate and the chain length above which the estimate is considered reliable. (Default value: 50)
pub fn auto_corr_time<C>(
    chain: C,
    min_win_size: Option<usize>,
    min_chain_len: Option<usize>,
) -> Option<f64>
where
    C: ExactSizeIterator<Item = f64> + Clone,
{
    let min_win_size = min_win_size.unwrap_or(5) as f64;
    let min_chain_len = min_chain_len.unwrap_or(50) as f64;

    let mean = chain.clone().sum::<f64>() / chain.len() as f64;

    let variance = chain
        .clone()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / chain.len() as f64;

    let mut estimate = 1.;

    for lag in 1..chain.len() {
        let auto_corr = chain
            .clone()
            .skip(lag)
            .zip(chain.clone())
            .map(|(lhs, rhs)| (lhs - mean) * (rhs - mean))
            .sum::<f64>()
            / chain.len() as f64
            / variance;

        estimate += 2. * auto_corr;

        if lag as f64 >= min_win_size * estimate {
            break;
        }
    }

    if chain.len() as f64 >= min_chain_len * estimate {
        Some(estimate)
    } else {
        None
    }
}

/// Determines how many iterations of the sampler are executed
///
/// Enables running the sampler until some condition based on
/// the samples collected so far is fulfilled, for example using
/// the [auto-correlation time][auto_corr_time].
///
/// The [`MinChainLen`] implementor provides a reasonable default.
///
/// It can also be used for progress reporting:
///
/// ```
/// use std::ops::ControlFlow;
///
/// use hammer_and_sample::{Params, Schedule};
///
/// struct FixedIterationsWithProgress {
///     done: usize,
///     todo: usize,
/// }
///
/// impl<P> Schedule<P> for FixedIterationsWithProgress
/// where
///     P: Params
/// {
///      fn next_step(&mut self, _chain: &[P]) -> ControlFlow<()> {
///         if self.done == self.todo {
///             eprintln!("100%");
///
///             ControlFlow::Break(())
///         } else {
///             self.done += 1;
///
///             if self.done % (self.todo / 100) == 0 {
///                 eprintln!("{}% ", self.done / (self.todo / 100));
///             }
///
///             ControlFlow::Continue(())
///         }
///     }
///
///     fn iterations(&self, _walkers: usize) -> Option<usize> {
///         Some(self.todo)
///     }
/// }
/// ```
pub trait Schedule<P>
where
    P: Params,
{
    /// The next step in the schedule given the current `chain`, either [continue][ControlFlow::Continue] or [break][ControlFlow::Break]
    fn next_step(&mut self, chain: &[P]) -> ControlFlow<()>;

    /// If possible, compute a lower bound for the number of iterations given the number of `walkers`
    fn iterations(&self, _walkers: usize) -> Option<usize> {
        None
    }
}

/// Runs the sampler until the given chain length is reached
pub struct MinChainLen(pub usize);

impl<P> Schedule<P> for MinChainLen
where
    P: Params,
{
    fn next_step(&mut self, chain: &[P]) -> ControlFlow<()> {
        if self.0 <= chain.len() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn iterations(&self, walkers: usize) -> Option<usize> {
        Some(self.0 / walkers)
    }
}

/// Runs the inner `schedule` after calling the given `callback`
///
/// ```
/// # use hammer_and_sample::{sample, MinChainLen, Model, Schedule, Serial, Stretch, WithProgress};
/// # use rand::SeedableRng;
/// # use rand_pcg::Pcg64Mcg;
/// #
/// # struct Dummy;
/// #
/// # impl Model for Dummy {
/// #     type Params = [f64; 1];
/// #
/// #     fn log_prob(&self, state: &Self::Params) -> f64 {
/// #         f64::NEG_INFINITY
/// #     }
/// # }
/// #
/// # let model = Dummy;
/// #
/// # let walkers = (0..100).map(|idx| {
/// #     let mut rng = Pcg64Mcg::seed_from_u64(idx);
/// #
/// #     ([0.], rng)
/// # });
/// #
/// let schedule = WithProgress {
///     schedule: MinChainLen(100_000),
///     callback: |chain: &[_]| eprintln!("{} %", 100 * chain.len() / 100_000),
/// };
///
/// let (chain, accepted) = sample(&model, &Stretch::default(), walkers, schedule, Serial);
/// ```
pub struct WithProgress<S, C> {
    /// The inner schedule which determines the number of iterations
    pub schedule: S,
    /// The callback which is executed after each iteration
    pub callback: C,
}

impl<P, S, C> Schedule<P> for WithProgress<S, C>
where
    P: Params,
    S: Schedule<P>,
    C: FnMut(&[P]),
{
    fn next_step(&mut self, chain: &[P]) -> ControlFlow<()> {
        (self.callback)(chain);

        self.schedule.next_step(chain)
    }

    fn iterations(&self, walkers: usize) -> Option<usize> {
        self.schedule.iterations(walkers)
    }
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
