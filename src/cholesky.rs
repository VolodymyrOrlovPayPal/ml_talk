use rand::prelude::*;
use core::arch::x86_64::*;

#[derive(Debug)]
pub struct NaiveMatrix {
    nrows: usize,
    ncols: usize,
    values: Vec<f64>
}

impl NaiveMatrix {

    pub fn from_vec(nrows: usize, ncols: usize, values: Vec<f64>) -> NaiveMatrix {
        NaiveMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values
        }
    }

    pub fn generate_positive_definite(nrows: usize, ncols: usize) -> NaiveMatrix {
        let m = NaiveMatrix::rand(nrows, ncols);
        m.dot(m.transpose())
    }

    pub fn transpose(&self) -> NaiveMatrix {
        let mut m = NaiveMatrix {
            ncols: self.nrows,
            nrows: self.ncols,
            values: vec![0f64; self.ncols * self.nrows]
        };
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                m.set(c, r, self.get(r, c));
            }
        }
        m

    }

    pub fn dot(&self, other: NaiveMatrix) -> NaiveMatrix {
        if self.ncols != other.ncols {
            panic!("Can only do dot product of matrices with similar inner dimentions!")
        }
        let mut m = NaiveMatrix {
            ncols: other.ncols,
            nrows: self.nrows,
            values: vec![0f64; self.nrows * other.ncols]
        };

        for r in 0..self.nrows {
            for c in 0..other.ncols {
                let mut sum = 0f64;
                for i in 0..self.ncols {
                    sum += self.get(r, i) * other.get(i, c);
                }
                m.set(r, c, sum);
            }
        }

        m
    }

    pub fn rand(nrows: usize, ncols: usize) -> NaiveMatrix {
        let mut rng = rand::thread_rng();
        let values: Vec<f64> = (0..nrows*ncols).map(|_| {
            rng.gen()
        }).collect();
        NaiveMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.values[col * self.nrows + row]
    }

    pub fn set(&mut self, row: usize, col: usize, v: f64) {
        self.values[col * self.nrows + row] = v;
    }

    pub fn naive_cholesky(&mut self) {
        let n_cols = self.ncols;
        let n_rows = self.ncols;
        assert!(n_cols == n_rows, "Can do Cholesky decomposition only for square matrix");
        for j in 0..n_rows {
            let mut d = 0f64;
            for k in 0..j {
                let mut s = 0f64;
                for i in 0..k {
                    s += self.get(k, i) * self.get(j, i);
                }
                s = (self.get(j, k) - s) / self.get(k, k);
                self.set(j, k, s);
                d = d + s * s;
            }
            d = self.get(j, j) - d;

            assert!(d > 0f64, "Matrix is not positive definite");

            self.set(j, j, d.sqrt());
        }
        
    }

    fn simd_128_f32() {
        unsafe{
            let a_values = _mm_set_ps(8.1239412, -931.20100, 5.531, -6.030100);
            let b_values = _mm_set_ps(9.0003, -20.202, 81325.20230, 195132.00999);
            println!("{:?}", _mm_mul_ps(a_values, b_values));
        }
    }

}

#[cfg(test)]
mod tests {

    use crate::cholesky::NaiveMatrix;

    #[test]
    fn naive_cholesky() {
        let mut m = NaiveMatrix::from_vec(3, 3, vec![0.68862408, 1.14997528, 0.5580459, 1.14997528, 1.98713229, 0.90468023, 0.5580459, 0.90468023, 0.46998922]);
        let expected = NaiveMatrix::from_vec(3, 3, vec![0.82, 1.38, 0.67, 1.14, 0.25, -0.10, 0.55, 0.90, 0.08]);
        m.naive_cholesky();        
        for c in 0..3 {
            for r in 0..3 {
                assert!((m.get(r, c) - expected.get(r, c)).abs() < 0.05);
            }
        }
    }

    #[test]
    fn rand() {
        let m = NaiveMatrix::rand(3, 3);
        for c in 0..3 {
            for r in 0..3 {
                assert!(m.get(r, c) != 0f64);
            }
        }
    }

    #[test]
    fn transpose() {
        let m = NaiveMatrix::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
        let expected = NaiveMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m_transposed = m.transpose();
        for c in 0..2 {
            for r in 0..2 {
                assert!(m_transposed.get(r, c) == expected.get(r, c));
            }
        }
    }

    #[test]
    fn generate_positive_definite() {
        let mut m = NaiveMatrix::generate_positive_definite(3, 3);
        println!("{:?}", m);
        let chol = m.naive_cholesky();
    }

    #[test]
    fn simd_test() {
        NaiveMatrix::simd_128_f32();        
    }
    
}