#ifndef TICTOC_HPP
#define TICTOC_HPP

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc
{
public:
    inline void tic()
    {
        ini = std::chrono::high_resolution_clock::now();
    }

    inline double toc()
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - ini).count() * 1000.0;
    }

    TicToc()
    {
        tic();
    }

private:
    std::chrono::high_resolution_clock::time_point ini;
};

#endif