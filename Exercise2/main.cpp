#include <iostream>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

// Function to solve linear system using PALU decomposition
VectorXd solvePALU(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

// Function to solve linear system using QR decomposition
VectorXd solveQR(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

// Function to compute relative error
double computeRelativeError(const VectorXd& x1, const VectorXd& x2) {
    return (x1 - x2).norm() / x1.norm();
}

int main() {
    // Define the matrices A and vectors b for the three systems
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    // Define the true solution for the systems
    VectorXd true_solution(2);
    true_solution << -1.0e+0, -1.0e+00;

    // Solve the systems and compute relative errors
    VectorXd x1_PALU = solvePALU(A1, b1);
    VectorXd x1_QR = solveQR(A1, b1);
    double relative_error1_PALU = computeRelativeError(x1_PALU, true_solution);
    double relative_error1_QR = computeRelativeError(x1_QR, true_solution);

    VectorXd x2_PALU = solvePALU(A2, b2);
    VectorXd x2_QR = solveQR(A2, b2);
    double relative_error2_PALU = computeRelativeError(x2_PALU, true_solution);
    double relative_error2_QR = computeRelativeError(x2_QR, true_solution);

    VectorXd x3_PALU = solvePALU(A3, b3);
    VectorXd x3_QR = solveQR(A3, b3);
    double relative_error3_PALU = computeRelativeError(x3_PALU, true_solution);
    double relative_error3_QR = computeRelativeError(x3_QR, true_solution);

    // Print the results
    cout << "System 1:\n";
    cout << "Relative Error (PALU): " << relative_error1_PALU << endl;
    cout << "Relative Error (QR): " << relative_error1_QR << endl;

    cout << "\nSystem 2:\n";
    cout << "Relative Error (PALU): " << relative_error2_PALU << endl;
    cout << "Relative Error (QR): " << relative_error2_QR << endl;

    cout << "\nSystem 3:\n";
    cout << "Relative Error (PALU): " << relative_error3_PALU << endl;
    cout << "Relative Error (QR): " << relative_error3_QR << endl;

    return 0;
}
