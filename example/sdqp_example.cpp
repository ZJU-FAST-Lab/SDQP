#include <iostream>

#include "sdqp/sdqp.hpp"
#include "sdqp/iosqp.hpp"

#include "tictoc.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    int m = 41 + 41 - 5;
    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Matrix<double, 3, 1> c;
    Eigen::Matrix<double, 3, 1> x;        // decision variables
    Eigen::Matrix<double, -1, 3> A(m, 3); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    Q << 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0;
    c << 1.2, 2.5, -10.0;

    A << 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, -1.0, -1.0,
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        0.0, 0.0, -1.0;
    b << 100000.0, 100000.0, 100000.0, -1.0, -2.0, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -2.1, -0.4, -1.1, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -2.1, -2.0, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -2.1, -0.4, -1.1, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -1.0, -2.0, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -2.1, -0.4, -1.1, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -2.1, -2.0, -3.0, -1.2, -2.1, -0.4, -1.1, -3.1, -0.1, -2.1, -0.4, -1.1, -3.0, -1.2, -2.1, -0.4, -1.1;

    int repeat = 100000;
    double minobj;
    TicToc t0;
    for (int i = 0; i < repeat; ++i)
    {
        minobj = sdqp::sdqp<3>(Q, c, A, b, x);
    }

    std::cout << "time: " << t0.toc() / repeat << std::endl;
    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << minobj << std::endl;
    std::cout << "optimal conv: " << (A * x - b).maxCoeff() << std::endl;

    IOSQP qp;

    Eigen::SparseMatrix<double> P = Q.sparseView();
    Eigen::SparseMatrix<double> AA = A.sparseView();
    Eigen::VectorXd cc = c;
    Eigen::VectorXd l = Eigen::VectorXd::Constant(m, -INFINITY);

    TicToc t1;
    qp.setMats(P, cc, AA, l, b, 1.0e-5, 1.0e-5);
    for (int i = 0; i < repeat; ++i)
    {
        osqp_setup(&(qp.pWork), qp.pData, qp.pSettings);
        qp.solve();
    }
    std::cout << "time*: " << t1.toc() / repeat << std::endl;
    std::cout << "optimal sol*: " << qp.getPrimalSol().transpose() << std::endl;
    std::cout << "optimal obj*: " << 0.5 * (Q * qp.getPrimalSol()).dot(qp.getPrimalSol()) + c.dot(qp.getPrimalSol()) << std::endl;
    std::cout << "optimal conv*: " << (A * qp.getPrimalSol() - b).maxCoeff() << std::endl;

    return 0;
}
