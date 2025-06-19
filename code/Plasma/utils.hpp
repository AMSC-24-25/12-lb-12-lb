#ifndef UTILS_HPP
#define UTILS_HPP

constexpr int Q = 9;

//Overload function to recover the index
    inline int INDEX(int x, int y, int i, int NX, int Q) {
        return i + Q * (x + NX * y);
    }
    inline int INDEX(int x, int y, int NX) {
        return x + NX * y;
    }

#endif // UTILS_HPP
