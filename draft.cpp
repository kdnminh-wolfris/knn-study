#include <cblas.h>
#include <stdio.h>

int main() {
  double A[6] = {1.0, 2.0,
                 1.0, -3.0,
                 4.0, -1.0};
  double B[6] = {1.0, 2.0, 1.0,
                 -3.0, 4.0,-1.0};
  double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 2, 1, A, 2, B, 3, 0, C, 3);

  for(int i=0; i<3; i++) {
    for (int j = 0; j < 3; ++j)
      printf("%lf ", C[i*3+j]);
    putchar('\n');
  }
}