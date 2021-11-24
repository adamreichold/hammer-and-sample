use rand::{
    distributions::{Distribution, Standard, Uniform},
    Rng,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelExtend, ParallelIterator};

pub trait Params: Send + Sync + Clone {
    const DIMENSION: usize;

    fn propose(&self, other: &Self, z_2: f64) -> Self;
}

impl<const N: usize> Params for [f64; N] {
    const DIMENSION: usize = N;

    fn propose(&self, other: &Self, z_2: f64) -> Self {
        let mut new = [0.; N];
        for i in 0..N {
            new[i] = other[i] + z_2 * (other[i] - self[i]);
        }
        new
    }
}

pub trait Model: Send + Sync {
    const SCALE: f64 = 2.;

    type Params: Params;

    fn log_prob(&self, state: &Self::Params) -> f64;
}

pub fn sample<M, W, R>(model: &M, walkers: W, iterations: usize) -> (Vec<M::Params>, usize)
where
    M: Model,
    W: Iterator<Item = (M::Params, R)>,
    R: Rng + Send + Sync,
{
    let mut walkers = walkers
        .map(|(state, rng)| Walker::new(model, state, rng))
        .collect::<Vec<_>>();

    assert!(!walkers.is_empty() && walkers.len() % 2 == 0);
    assert!(walkers.len() >= 2 * M::Params::DIMENSION);

    let mut chain = Vec::with_capacity(walkers.len() * iterations);

    let half = walkers.len() / 2;
    let (lower_half, upper_half) = walkers.split_at_mut(half);

    let random_index = Uniform::new(0, half);

    for _ in 0..iterations {
        extend_chain(&mut chain, lower_half, |walker| {
            let other = &upper_half[random_index.sample(&mut walker.rng)];

            walker.move_(model, other);

            walker.state.clone()
        });

        extend_chain(&mut chain, upper_half, |walker| {
            let other = &lower_half[random_index.sample(&mut walker.rng)];

            walker.move_(model, other);

            walker.state.clone()
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

    fn move_(&mut self, model: &M, other: &Self) {
        let z = (M::SCALE - 1.) * gen_unit(&mut self.rng) + 1.;
        let z_2 = z * z / M::SCALE;

        let new_state = self.state.propose(&other.state, z_2);
        let new_log_prob = model.log_prob(&new_state);

        let log_prob_diff =
            (M::Params::DIMENSION - 1) as f64 * z_2.ln() + new_log_prob - self.log_prob;

        if log_prob_diff > gen_unit(&mut self.rng).ln() {
            self.state = new_state;
            self.log_prob = new_log_prob;
            self.accepted += 1;
        }
    }
}

#[cfg(feature = "rayon")]
fn extend_chain<M, R, U>(chain: &mut Vec<M::Params>, walkers: &mut [Walker<M, R>], update: U)
where
    M: Model,
    R: Send,
    U: Fn(&mut Walker<M, R>) -> M::Params + Send + Sync,
{
    chain.par_extend(walkers.par_iter_mut().map(update));
}

#[cfg(not(feature = "rayon"))]
fn extend_chain<M, R, U>(chain: &mut Vec<M::Params>, walkers: &mut [Walker<M, R>], update: U)
where
    M: Model,
    U: Fn(&mut Walker<M, R>) -> M::Params,
{
    chain.extend(walkers.iter_mut().map(update));
}

fn gen_unit<R>(rng: &mut R) -> f64
where
    R: Rng,
{
    Distribution::<f64>::sample(&Standard, rng)
}
