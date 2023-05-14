#include<iostream>
#include<fstream>
#include<omp.h>
#include<sys/time.h>
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
        #pragma omp for
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
	timeval start,finish;




    gettimeofday(&start,NULL);// Start Time
    OpenMPLU();
	gettimeofday(&finish,NULL);// End Time
    cout<<((finish.tv_sec-start.tv_sec)*1000000.0+finish.tv_usec-start.tv_usec)/1000.0<<endl;



}

