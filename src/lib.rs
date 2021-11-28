#[cfg(feature = "python-bindings")]
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rand::{
    distributions::{Distribution, Standard, Uniform},
    Rng,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelExtend, ParallelIterator};

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

#[cfg(feature = "python-bindings")]
#[pymodule]
#[pyo3(name = "hammer_and_sample")]
fn init(_py: Python, module: &PyModule) -> PyResult<()> {
    use numpy::{ndarray::Zip, IntoPyArray, PyArray1, PyArray2};
    use pyo3::{PyAny, Python};
    use rand::{rngs::SmallRng, SeedableRng};

    #[pyfn(module)]
    #[pyo3(name = "sample")]
    fn wrapper<'py>(
        py: Python<'py>,
        log_prob: &'py PyAny,
        guess: Vec<&'py PyArray1<f64>>,
        seed: u64,
        iterations: usize,
    ) -> (&'py PyArray2<f64>, usize) {
        #[derive(Clone, Copy)]
        struct PyParams<'py>(&'py PyArray1<f64>);

        impl Params for PyParams<'_> {
            fn dimension(&self) -> usize {
                self.0.len()
            }

            fn propose(&self, other: &Self, z: f64) -> Self {
                Self(
                    Zip::from(self.0.readonly().as_array())
                        .and(other.0.readonly().as_array())
                        .map_collect(|self_, other| other - z * (other - self_))
                        .into_pyarray(self.0.py()),
                )
            }
        }

        unsafe impl Send for PyParams<'_> {}

        unsafe impl Sync for PyParams<'_> {}

        struct PyModel<'py>(&'py PyAny);

        impl<'py> Model for PyModel<'py> {
            type Params = PyParams<'py>;

            fn log_prob(&self, state: &Self::Params) -> f64 {
                self.0.call1((state.0,)).unwrap().extract().unwrap()
            }
        }

        unsafe impl Send for PyModel<'_> {}

        unsafe impl Sync for PyModel<'_> {}

        let mut rng = SmallRng::seed_from_u64(seed);

        let walkers = guess.into_iter().map(|state| {
            let rng = SmallRng::from_rng(&mut rng).unwrap();

            (PyParams(state), rng)
        });

        let (chain, accepted) = sample(&PyModel(log_prob), walkers, iterations, &Serial);

        unsafe {
            let flat_chain = PyArray2::<f64>::new(py, (chain.len(), chain[0].dimension()), false);

            for (i, sample) in chain.into_iter().enumerate() {
                for (j, param) in sample.0.readonly().iter().unwrap().enumerate() {
                    *flat_chain.uget_mut((i, j)) = *param;
                }
            }

            (flat_chain, accepted)
        }
    }

    Ok(())
}
