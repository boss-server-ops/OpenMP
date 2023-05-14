#include<iostream>
#include<Windows.h>
#include<fstream>
#include<omp.h>
using namespace std;
const int N = 2000;
const int NUM_THREADS=7;
float m[N][N];

void OpenMPLU() {
    int i,j,k;
    float tmp;
    bool parallel=true;
    #pragma omp parallel if(parallel),num_threads(NUM_THREADS),private(i,j,k,tmp)
	for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
        tmp=m[k][k];
		for (j = k + 1; j < N; j++) {
			m[k][j] = m[k][j] / tmp;
		}
		m[k][k] = 1.0;
        }
        #pragma omp for schedule(dynamic,10)
		for (i = k+1; i < N; i++) {
            tmp=m[i][k];
			for ( j= k+1; j < N; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
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
