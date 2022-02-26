#include <iostream>

#include "sdqp/sdqp.hpp"
#include "sdqp/iosqp.hpp"

#include "tictoc.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    int m = 2 * 5;
    Eigen::Matrix<double, 5, 1> x;        // decision variables
    Eigen::Matrix<double, -1, 5> A(m, 5); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    A << 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0, 0.0, 0.0,
        1.0, -1.0, 0.0, 0.0, 2.0,
        0.0, 0.0, -1.0, 2.0, 0.0,
        0.0, 0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, -1.0;
    b << 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, -1.0, -2.0, -3.0, -4.0, -5.0;

    int repeat = 100000;
    double minobj;
    TicToc t0;
    for (int i = 0; i < repeat; ++i)
    {
        minobj = sdqp::sdmn<5>(A, b, x);
    }

    std::cout << "time: " << t0.toc() / repeat << std::endl;
    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << minobj << std::endl;
    std::cout << "optimal conv: " << (A * x - b).maxCoeff() << std::endl;

    IOSQP qp;

    Eigen::SparseMatrix<double> P = Eigen::MatrixXd::Identity(5, 5).sparseView();
    Eigen::VectorXd q = Eigen::VectorXd::Zero(5);
    Eigen::SparseMatrix<double> AA = A.sparseView();
    Eigen::VectorXd l = Eigen::VectorXd::Constant(m, -INFINITY);

    TicToc t1;
    qp.setMats(P, q, AA, l, b, 1.0e-7, 1.0e-7);
    for (int i = 0; i < repeat; ++i)
    {
        qp.solve();
    }
    std::cout << "time*: " << t1.toc() / repeat << std::endl;
    std::cout << "optimal sol*: " << qp.getPrimalSol().transpose() << std::endl;
    std::cout << "optimal obj*: " << qp.getPrimalSol().norm() << std::endl;
    std::cout << "optimal conv*: " << (A * qp.getPrimalSol() - b).maxCoeff() << std::endl;

    return 0;
}