extern crate ml_talk;
use criterion::{criterion_group, criterion_main, Criterion};
use ml_talk::cholesky::NaiveMatrix;

pub fn naive_cholesky(c: &mut Criterion) {
    c.bench_function("Naive Cholesky", |b| b.iter(|| NaiveMatrix::generate_positive_definite(16, 16).naive_cholesky()));
}

criterion_group!(benches, naive_cholesky);
criterion_main!(benches);