#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../include/matrix.h"

namespace
{
bool near(double a, double b)
{
    return std::abs(a - b) < 1e-9;
}
}

int main()
{
    {
        Matrix m(2, 3);
        assert(m.rows == 2);
        assert(m.cols == 3);
        assert(m.data.size() == 6);

        m(0, 0) = 1.0;
        m(0, 1) = 2.0;
        m(0, 2) = 3.0;
        m(1, 0) = 4.0;
        m(1, 1) = 5.0;
        m(1, 2) = 6.0;

        Matrix t = m.transpose();
        assert(t.rows == 3);
        assert(t.cols == 2);
        assert(near(t(0, 0), 1.0));
        assert(near(t(1, 0), 2.0));
        assert(near(t(2, 1), 6.0));
    }

    {
        Matrix a(2, 3);
        a(0, 0) = 1.0;
        a(0, 1) = 2.0;
        a(0, 2) = 3.0;
        a(1, 0) = 4.0;
        a(1, 1) = 5.0;
        a(1, 2) = 6.0;

        Matrix b(3, 2);
        b(0, 0) = 7.0;
        b(0, 1) = 8.0;
        b(1, 0) = 9.0;
        b(1, 1) = 10.0;
        b(2, 0) = 11.0;
        b(2, 1) = 12.0;

        Matrix c = a.matmul(b);
        assert(c.rows == 2);
        assert(c.cols == 2);
        assert(near(c(0, 0), 58.0));
        assert(near(c(0, 1), 64.0));
        assert(near(c(1, 0), 139.0));
        assert(near(c(1, 1), 154.0));
    }

    {
        Matrix a(2, 2, 1.5);
        Matrix b(2, 2, 2.0);
        Matrix c = a.add(b);
        assert(near(c(0, 0), 3.5));
        assert(near(c(1, 1), 3.5));
    }

    {
        Matrix m(2, 3, 1.0);
        Matrix y = m.add_row_vector({10.0, 20.0, 30.0});
        assert(near(y(0, 0), 11.0));
        assert(near(y(0, 2), 31.0));
        assert(near(y(1, 1), 21.0));
    }

    {
        Matrix m(1, 4);
        m(0, 0) = -2.0;
        m(0, 1) = 0.0;
        m(0, 2) = 3.0;
        m(0, 3) = -4.0;

        Matrix y = m.relu();
        assert(near(y(0, 0), 0.0));
        assert(near(y(0, 1), 0.0));
        assert(near(y(0, 2), 3.0));
        assert(near(y(0, 3), 0.0));
    }

    {
        Matrix m(2, 3);
        m(0, 0) = 1.0;
        m(0, 1) = 2.0;
        m(0, 2) = 3.0;
        m(1, 0) = 1000.0;
        m(1, 1) = 1000.0;
        m(1, 2) = 1000.0;

        Matrix y = m.softmax_rows();
        assert(y.rows == 2);
        assert(y.cols == 3);

        double row0_sum = y(0, 0) + y(0, 1) + y(0, 2);
        double row1_sum = y(1, 0) + y(1, 1) + y(1, 2);
        assert(near(row0_sum, 1.0));
        assert(near(row1_sum, 1.0));
        assert(y(0, 2) > y(0, 1));
        assert(y(0, 1) > y(0, 0));
        assert(near(y(1, 0), 1.0 / 3.0));
        assert(near(y(1, 1), 1.0 / 3.0));
        assert(near(y(1, 2), 1.0 / 3.0));
    }

    std::cout << "All matrix tests passed!" << std::endl;

    return 0;
}
