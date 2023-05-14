#include<iostream>
#include<fstream>
#include<omp.h>
#include<arm_neon.h>
#include<sys/time.h>
using namespace std;
const int N = 2000;
const int NUM_THREADS=7;
float m[N][N];

void OpenMPLU() {
    int i,j,k;
    float tmp;
    bool parallel=true;
    float32x4_t vt, va, vb, vc;
    #pragma omp parallel if(parallel),num_threads(NUM_THREADS),private(i,j,k,vt,va,vb,vc)
	for (k = 0; k < N; k++)
    {
       #pragma omp single
        {
            vt = vld1q_dup_f32(&m[k][k]);
            int j=0;
            for (j = k + 1; j+4 <= N; j+=4)
            {
                va = vld1q_f32(&m[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&m[k][j], va);
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

            vt = vld1q_dup_f32(&m[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
			{
				va = vld1q_f32(&m[k][j]);
				vb = vld1q_f32(&m[i][j]);
				vc = vmulq_f32(vt, va);
				vb = vsubq_f32(vb, vc);
				vst1q_f32(&m[i][j], vb);
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
	timeval start,finish;




    gettimeofday(&start,NULL);// Start Time
    OpenMPLU();
	gettimeofday(&finish,NULL);// End Time
    cout<<((finish.tv_sec-start.tv_sec)*1000000.0+finish.tv_usec-start.tv_usec)/1000.0<<endl;



}

