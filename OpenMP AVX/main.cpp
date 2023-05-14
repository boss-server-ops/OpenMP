#include<iostream>
#include<Windows.h>
#include<fstream>
#include<omp.h>
#include <immintrin.h>
using namespace std;
const int N = 2000;
const int NUM_THREADS=4;
float m[N][N];

void OpenMPLU() {
    int i,j,k;
    float tmp;
    bool parallel=true;
    __m256_u vt, va, vb, vc;
    #pragma omp parallel if(parallel),num_threads(NUM_THREADS),private(i,j,k,tmp,vt,va,vb,vc)
	for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            float tmp[8]={m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k],m[k][k]};
            vt=_mm256_loadu_ps(tmp);
            for (j = k + 1; j+8 <= N; j+=8)
            {
                va = _mm256_loadu_ps(&m[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&m[k][j], va);
            }
            while (j < N) {
                m[k][j] = m[k][j] / m[k][k];
                j++;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for
        for(int i=k+1;i<N;i++)
        {

            float tmp2[8]={m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k],m[i][k]};
            vt=_mm256_loadu_ps(tmp2);
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                va = _mm256_loadu_ps(&m[k][j]);
				vb = _mm256_loadu_ps(&m[i][j]);
				vc = _mm256_mul_ps(vt, va);
				vb = _mm256_sub_ps(vb, vc);
				_mm256_storeu_ps(&m[i][j], vb);
            }
            while (j < N) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                j++;
            }
            m[i][k] = 0;
        }
	}
}
int main()
{
    ifstream infile("F:\\example.txt");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			char c;
			infile >> m[i][j];

		}
	}
    infile.close();
    cout<<m[0][1]<<endl;
	long long head, tail, freq; // timers

	 // similar to CLOCKS_PER_SEC
		 QueryPerformanceFrequency((LARGE_INTEGER*)  & freq);
	 // start time
		 QueryPerformanceCounter((LARGE_INTEGER*)& head);
	OpenMPLU();
	QueryPerformanceCounter((LARGE_INTEGER*) & tail);
	 cout << "serialCol: " << (tail-head) * 1000.0 / freq << "ms" << endl;



}
