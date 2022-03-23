#include <iostream>

template<typename T>
T* choose_pivot(T* first, T* last) {
    return first + ((last - first) >> 1);
}

template<typename T>
T* partition(T* first, T* last) {
    std::swap(*choose_pivot(first, last), *(last - 1));
    T* pivot = first;
    for (; first < last - 1; ++first)
        if (*first <= *(last - 1)) {
            std::swap(*first, *pivot);
            ++pivot;
        }
    std::swap(*pivot, *(last - 1));
    return pivot;
}

template<typename T>
void sort(T* first, T* last, int k) {
    if (first >= last) return;
    if (k == 0) return;
    if (first == last - 1) return;
    if (first == last - 2) {
        if (*first > *(last - 1))
            std::swap(*first, *(last - 1));
        return;
    }
    T* pivot = partition(first, last);
    sort(first, pivot, std::min(k, int(pivot - first)));
    sort(pivot + 1, last, std::max(0, k - int(pivot - first) - 1));
}

int a[10] = {1, 4, 2, 6, 3, 7, 9, 8, 10, 5};

int main() {
    sort(a, a + 10, 2);
    for (int i = 0; i < 10; ++i)
        std::cout << a[i] << ' ';
}