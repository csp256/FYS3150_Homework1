#include <iostream>
#include <armadillo>
#include <cfloat>
#include <time.h>
#include <matrixSolvers.hpp>

using namespace std;
using namespace arma;

int main() {
    const int n = pow(10,5);
    std::cout << "n: 10^" << log10(n) << endl;
    const float h  = 1.0f / ((float) n + 1.0f); // h is the spacing between discretized points
    const float h2 = h*h;
    float x; // x is a spatial coordinate
    Col<float> b(n), u(n), v(n); // as per assignment
    clock_t start, finish;

    // Find the RHS of the Poisson eq 'b' (up to a constant multiple)
    // and the LHS's analytic solution 'u'.
    for (int i=0; i<n; ++i) {
        x = (float) (i+1) * h;
        b(i) = h2*100.0f * exp(-10.0f * x);
        u(i) = 1.0f - (1.0f - exp(-10.0f))*x - exp(-10.0f*x);
    }

    // Calculate discrete solution (and time it).
    start = clock();
    v = poissonSolver(b);
    finish = clock();
    cout << "Time: " << finish-start << "us" << endl;

    // Find common log of the solver's largest deviation from the analytic solution.
    float maxError = -FLT_MAX;
    float temp;
    for (int i=0; i<n; ++i) {
        temp = abs(v(i) - u(i));
        if (maxError < temp) {
            maxError = temp;
        }
    }
    maxError = log10(maxError);
    cout << "maxError: " << maxError << endl << endl;

    // Compare with Armadillo.
    mat A(n,n);
    A.zeros();
    A.diag( 0)  +=  2.0f;
    A.diag(-1)  += -1.0f;
    A.diag( 1)  += -1.0f;
    colvec B(n), X(n);
    for (int i=0; i<n; ++i) {
        B(i) = b(i);
    }
    start = clock();
    X = solve(A, B);
    finish = clock();
    cout << "Time: " << (finish-start)/1000 << "ms" << endl;
    maxError = -FLT_MAX;
    for (int i=0; i<n; ++i) {
        temp = abs(X(i) - u(i));
        if (maxError < temp) {
            maxError = temp;
        }
    }
    cout << "Armadillo max deviation:" << endl << log10(maxError) << endl;

    return 0;
}
