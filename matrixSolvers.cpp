#include <armadillo>

using namespace arma;

// This assumes a tridiagonal symmetric toeplitz matrix,
// where the main diagonal is 2 and the off diagonals are -1.
// This is used to solve Poisson's equation w/ Dirichlet BC's.
// The LU factorization of the matrix is embarassingly parallel.
// This is an optimized form of the Thomas algorithm.
// It can be adapted to work with other tridiagonal, symmetric,
// Toeplitz matrices by finding a closed form of a recurrence
// relation of the terms in the L and U matrices.
//
// Instruction      serial      parallelizable     total
//
// FMA              n                               n
// ADD              n           n (0?)              2n (maybe n?)
// MULTIPLY         n                               n
// DIVIDE                       n                   n
// ------------------------------------------------------
//                  3n          2n                  5n < 6n
Col<float> poissonSolver(Col<float> q) {
    const int n = q.n_elem;
    Col<float> d(n), l(n); // d is for main diagonal in U, l is lower diagonal in L
    Col<float> y(n), x(n); // y is a temporary vector, and x is the final result.

    // LU decomposition
    l(0) = 0; // actually, it is just undefined!
    float I0, I1; // where I# represents how far ahead of the iterator 'i' the variable is
    I0 = 1.0f; I1 = 2.0f;
    for (int i=1; i<n; i++) {
        l(i)    = I0 / I1;
        d(i-1)  = l(i); // This is the INVERSE of 'd'! We store the reciprocals.

        I0  = I1;
        I1 += 1.0f; // Is it cheaper to typecast the int to a float to avoid this add? (We need an int to index l & d.)
    }
    d(n-1) = I0 / I1; // The loop above would miss the last element of d.


    // Forward substitution.
    y(0) = q(0);
    for (int i=1; i<n; ++i) {
        y(i) = fma(l(i), y(i-1), q(i)); // Remember, we previously stored the NEGATIVE of 'l'!
    }

    // Backward substitution.
    x(n-1) = y(n-1) * d(n-1); // We stored the INVERSE of 'd'!
    for (int i=n-2; 0<=i; --i) {
        x(i) = (x(i+1) + y(i)) * d(i);
    }

    return x;
}


// Solves Ax=q where 'b' is the main diagonal and 'a' is the off diagonal
// Assumes input is benign (positive definite or positive semidefinite)
// This combines LU factorization and the Thomas algorithm.
// This function is not used in homework assignment 1.
Col<float> tridiagonalSymmetricToeplitzSolver( float a, float b, Col<float> q) {
    const int n = q.n_elem;
    const float negative_a = -a;
    Col<float> d(n), l(n); // See further notes in poissonSolver()
    Col<float> y(n), x(n);

    // LU decomposition
    d(0) = 1.0f / b; // This is the INVERSE of d(0)! We store the reciprocals.
    l(0) = 0; // actually, it is just undefined!
    for (int i=1; i<n; i++) {
        l(i) = negative_a * d(i-1); // This stores the NEGATIVE of 'l'!
        d(i) = 1.0f / fma(a, l(i), b); // This stores the INVERSE of 'd'!
    }

    // Forward substitution.
    y(0) = q(0);
    for (int i=1; i<n; ++i) {
        y(i) = fma(l(i), y(i-1), q(i)); // Remember, we previously stored the NEGATIVE of 'l'!
    }

    // Backward substitution.
    x(n-1) = y(n-1) * d(n-1); // We stored the INVERSE of 'd'!
    for (int i=n-2; 0<=i; --i) {
        x(i) = fma(negative_a, x(i+1), y(i)) * d(i); // Remember, we previously stored the inverse of 'd', so we multiply instead of dividing!
    }

    return x;
}
