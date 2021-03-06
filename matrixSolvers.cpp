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
// ADD              n                               n
// MULTIPLY         n                               n
// DIVIDE                       n                   n
// ------------------------------------------------------
//                  3n          n                   4n FLOPs
Col<float> poissonSolver(Col<float> q) {
    const int n = q.n_elem;
    Col<float> d(n); // The d vector occurs twice in the LU factorization, up to negation and multiplicative inversion.
    Col<float> y(n), x(n); // y is a temporary vector, and x is the final result.

    // Forward substitution.
    {
        int prev=0,curr=1; // Better to let these fall out of scope and let compiler optimize backwards loop w/ variables of same name
        y(0) = q(0);
        for (float fcurr = 1.0f, fnext = 2.0f; curr<n; ++prev, ++curr) { // Compiler seems to optimize better with float declerations here
            d(prev) = (fcurr) / (fnext); // This is a hell of a lot cheaper than normal LU decomp.
            fcurr = (fnext); // Halves the number of float promotes necessary.
            fnext = (float)(curr+2); // You save nearly 60% runtime by adding 2 here instead of adding 1 after ++curr !!

            y(curr) = fma(d(prev), y(prev), q(curr)); // Remember, we previously stored the NEGATIVE of 'l'!
        }
        d(prev) = (float)(curr) / (float)(curr+1); // Cost of the float promotes vanishes
    }

    // Backward substitution.
    x(n-1) = y(n-1) * d(n-1); // We stored the INVERSE of 'd'!
    for (int curr=n-2, prev=n-1; 0<=curr; --prev,--curr) {
        x(curr) = (x(prev) + y(curr)) * d(curr);
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
    Col<float> d(n), l(n); // 'd' is the U main diagonal, and 'l' is the L off diagonal.
    Col<float> y(n), x(n); // 'y' stores intermediate results, and 'x' is the final result.

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
