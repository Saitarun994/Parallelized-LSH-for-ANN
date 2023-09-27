#include<stdio.h>
#include<string.h>
#include<random>
#include<string>
#include<algorithm>
#include<set>
#include<random>
#include<cmath>

std::mt19937 generator(123456789);
std::normal_distribution<double> dist(0.0,1.0);
const int max_iter = 100;

double cosine_similarity(const std::vector<double>& u,const std::vector<double>& v)
{
  double ip = 0;
  double sumu2 = 0;
  double sumv2 = 0;
  for(int i = 0;i < u.size();++i)
  {
    ip += u[i] * v[i];
    sumu2 += u[i] * u[i];
    sumv2 += v[i] * v[i];
  }
  return ip / sqrt(sumu2) / sqrt(sumv2);
}

void dump_func(int cas,int D,int N,int M,double T)
{
	printf("dumping case %d\n",cas);
	std::string file_name = "sample" + std::to_string(cas) + ".in";
	FILE* fp = fopen(file_name.c_str(),"w");
	fprintf(fp,"%d %d %d %f\n",D,N,M,T);
  std::vector<std::vector<double>> data;
  for(int i = 0;i < N;++i)
  {
    std::vector<double> tmp;
    for(int j = 0;j < D;++j)
      tmp.push_back(dist(generator));
    data.push_back(tmp);
    fprintf(fp,"%f",tmp[0]);
    for(int j = 1;j < D;++j)
      fprintf(fp," %f",tmp[j]);
    fprintf(fp,"\n");
  }

  std::uniform_int_distribution<int> distint(0,N - 1);
  for(int i = 0;i < M;++i)
  {
    double eps = 1e-1;
    bool found = false;
    while(!found){
      std::normal_distribution<double> disteps(0.0,eps);
      for(int iter = 0;iter < max_iter;++iter)
      {
        int idx = distint(generator);
        std::vector<double> q = data[idx];
        for(int j = 0;j < D;++j)
          q[j] += disteps(generator);
        if(cosine_similarity(q,data[idx]) > T)
        {
          found = true;
          fprintf(fp,"%f",q[0]);
          for(int j = 1;j < D;++j)
            fprintf(fp," %f",q[j]);
          fprintf(fp,"\n");
          break;
        }
      }
      eps /= 2;
    }
  }
	fclose(fp);
}

int main()
{
  
  dump_func(1,2   ,10      ,5    ,0.9);
  dump_func(2,128 ,500000  ,10000,0.9);
  dump_func(3,128 ,1000000 ,10000,0.9);
  dump_func(4,128 ,1000000 ,10000,0.95);
  dump_func(5,256 ,1000000 ,10000,0.9);
  /*
  dump_func(6,256 ,1000000 ,10000,0.95);
  dump_func(7,512 ,500000  ,10000,0.9);
  dump_func(8,512 ,500000  ,10000,0.95);
  dump_func(9,512 ,1000000 ,10000,0.9);
  dump_func(10,512,1000000 ,10000,0.95);
  */
  return 0;
}

