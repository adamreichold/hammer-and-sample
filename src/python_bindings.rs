use numpy::{ndarray::Zip, IntoPyArray, PyArray1, PyArray2};
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Py, PyAny, PyResult, Python};
use rand::{rngs::SmallRng, SeedableRng};

use super::{sample, Model, Parallel, Params, Serial};

#[pymodule]
#[pyo3(name = "hammer_and_sample")]
fn init(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(wrapper, module)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "sample")]
fn wrapper<'py>(
    py: Python<'py>,
    log_prob: &'py PyAny,
    guess: Vec<&'py PyArray1<f64>>,
    seed: u64,
    iterations: usize,
    execution: Option<&str>,
) -> (&'py PyArray2<f64>, usize) {
    match execution {
        None | Some("serial") => wrapper_serial(py, log_prob, guess, seed, iterations),
        #[cfg(feature = "rayon")]
        Some("parallel") => wrapper_parallel(py, log_prob, guess, seed, iterations),
        Some(exection) => panic!("Unknown execution strategy '{}' given", exection),
    }
}

fn wrapper_serial<'py>(
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

    let model = PyModel(log_prob);

    let mut rng = SmallRng::seed_from_u64(seed);

    let walkers = guess.into_iter().map(|state| {
        let rng = SmallRng::from_rng(&mut rng).unwrap();

        (PyParams(state), rng)
    });

    let (chain, accepted) = sample(&model, walkers, iterations, &Serial);

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

#[cfg(feature = "rayon")]
fn wrapper_parallel<'py>(
    py: Python<'py>,
    log_prob: &'py PyAny,
    guess: Vec<&'py PyArray1<f64>>,
    seed: u64,
    iterations: usize,
) -> (&'py PyArray2<f64>, usize) {
    #[derive(Clone)]
    struct PyParams(Py<PyArray1<f64>>);

    impl Params for PyParams {
        fn dimension(&self) -> usize {
            Python::with_gil(|py| self.0.as_ref(py).len())
        }

        fn propose(&self, other: &Self, z: f64) -> Self {
            Python::with_gil(|py| {
                Self(
                    Zip::from(self.0.as_ref(py).readonly().as_array())
                        .and(other.0.as_ref(py).readonly().as_array())
                        .map_collect(|self_, other| other - z * (other - self_))
                        .into_pyarray(py)
                        .into(),
                )
            })
        }
    }

    unsafe impl Send for PyParams {}

    unsafe impl Sync for PyParams {}

    struct PyModel(Py<PyAny>);

    impl Model for PyModel {
        type Params = PyParams;

        fn log_prob(&self, state: &Self::Params) -> f64 {
            Python::with_gil(|py| {
                self.0
                    .as_ref(py)
                    .call1((state.0.clone(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            })
        }
    }

    unsafe impl Send for PyModel {}

    unsafe impl Sync for PyModel {}

    let model = PyModel(log_prob.into());

    let mut rng = SmallRng::seed_from_u64(seed);

    let walkers = guess
        .into_iter()
        .map(|state| {
            let rng = SmallRng::from_rng(&mut rng).unwrap();

            (PyParams(state.into()), rng)
        })
        .collect::<Vec<_>>();

    let (chain, accepted) =
        py.allow_threads(move || sample(&model, walkers.into_iter(), iterations, &Parallel));

    unsafe {
        let flat_chain = PyArray2::<f64>::new(py, (chain.len(), chain[0].dimension()), false);

        for (i, sample) in chain.into_iter().enumerate() {
            for (j, param) in sample.0.as_ref(py).readonly().iter().unwrap().enumerate() {
                *flat_chain.uget_mut((i, j)) = *param;
            }
        }

        (flat_chain, accepted)
    }
}
