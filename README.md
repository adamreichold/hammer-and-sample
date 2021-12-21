# hammer-and-sample

[![crates.io](https://img.shields.io/crates/v/hammer-and-sample.svg)](https://crates.io/crates/hammer-and-sample)
[![docs.rs](https://docs.rs/hammer-and-sample/badge.svg)](https://docs.rs/hammer-and-sample)
[![github.com](https://github.com/adamreichold/hammer-and-sample/actions/workflows/test.yaml/badge.svg)](https://github.com/adamreichold/hammer-and-sample/actions/workflows/ci.yaml)

A simplistic MCMC sampler implementing the affine-invariant ensemble sampling of [emcee](https://emcee.readthedocs.io/) with serial execution and optionally with parallel execution based on Rayon.

The implementation is relatively efficient, for example computing 1000 iterations of 100 walkers using the hierarchical model from [`hierarchical.rs`](tests/hierarchical.rs) takes approximately 1 min using `emcee` and `multiprocessing` versus 50 ms using this crate and Rayon, running on 8 hardware threads in both cases.
