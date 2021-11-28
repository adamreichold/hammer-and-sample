use rand::{
    distributions::{Distribution, Standard, Uniform},
    Rng,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelExtend, ParallelIterator};

#[cfg(feature = "python_bindings")]
mod python_bindings;

pub trait Params: Send + Sync + Clone {
    fn dimension(&self) -> usize;

    fn propose(&self, other: &Self, z: f64) -> Self;
}

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

pub trait Model: Send + Sync {
    const SCALE: f64 = 2.;

    type Params: Params;

    fn log_prob(&self, state: &Self::Params) -> f64;
}

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

pub struct Walker<M, R>
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

pub trait Execution {
    fn extend_chain<M, R, U>(
        &self,
        chain: &mut Vec<M::Params>,
        walkers: &mut [Walker<M, R>],
        update: U,
    ) where
        M: Model,
        R: Send + Sync,
        U: Fn(&mut Walker<M, R>) -> M::Params + Send + Sync;
}

pub struct Serial;

impl Execution for Serial {
    fn extend_chain<M, R, U>(
        &self,
        chain: &mut Vec<M::Params>,
        walkers: &mut [Walker<M, R>],
        update: U,
    ) where
        M: Model,
        R: Send + Sync,
        U: Fn(&mut Walker<M, R>) -> M::Params + Send + Sync,
    {
        chain.extend(walkers.iter_mut().map(update));
    }
}

#[cfg(feature = "rayon")]
pub struct Parallel;

#[cfg(feature = "rayon")]
impl Execution for Parallel {
    fn extend_chain<M, R, U>(
        &self,
        chain: &mut Vec<M::Params>,
        walkers: &mut [Walker<M, R>],
        update: U,
    ) where
        M: Model,
        R: Send + Sync,
        U: Fn(&mut Walker<M, R>) -> M::Params + Send + Sync,
    {
        chain.par_extend(walkers.par_iter_mut().map(update));
    }
}
