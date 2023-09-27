#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iterator>
#include <fstream>
#include <algorithm>
#include<stdlib.h>
#include<time.h>
#include <iostream>
#include <string.h>
#include <unordered_map>
#include<string>
#include <cmath>
#include<climits>
#include <sstream>
#include <cstdlib>

using namespace std;
using namespace std::chrono;
std::vector<std::vector<double>> base;  
std::vector<std::vector<double>> query;


/*
 * Reads the entire file line by line and 
 * stores each line in a string vector
 */
std::vector<string> read_file(ifstream &fin)  
{
  std::vector<string> lines;
  string line;
  while(getline(fin,line,'\n'))
  {
    lines.push_back(line);
  }
  return lines;
}

/*
 * Reads in lines from str vector and 
 * assign values to matrix in parallel
 */
std::vector<std::vector<double>> read_matrix(vector<string> lines,int row,int col,int start_idx) 
{
  std::vector<std::vector<double>> ret;
  ret.resize(row);
  #pragma omp parallel
  {
    #pragma omp for 
    for(int i=0;i<row;i++)
    {   
      ret[i].resize(col);
      std::stringstream ss(lines[start_idx+i]);
      std::istream_iterator<std::string> begin(ss);
      std::istream_iterator<std::string> end;
      std::vector<std::string> numbers(begin, end);  
            
      for(int j=0;j<col;j++)
      {
        double val = atof(numbers[j].c_str());
        ret[i][j]=val;
      }
    }
  }
  return ret;
}


double cosine_similarity(const std::vector<double>& u,const std::vector<double>& v) // <<<Original>>>
{
  double ip = 0;
  double sumu2 = 0;
  double sumv2 = 0;
  for(int i = 0;i < u.size();++i)
  { 
      ip += u[i] * v[i];
      sumu2 += u[i]*u[i];
      sumv2 += v[i]*v[i];
  }
  return ip / (sqrt(sumu2 * sumv2));;
}


/* 
*   Used to create hyperplanes 
*   to divide the 2d vector space randomly
*/
std::vector<std::vector<double>> hyperplane_generator(int d,int n_bits)
{
    std::vector<std::vector<double>> ret;
    ret.resize(d);
    for(int i=0;i<d;i++)
    { ret[i].resize(n_bits);
     for(int j=0;j<n_bits;j++)
     { 
        ret[i][j] = (double) rand()/RAND_MAX - 0.5;
     }
    }
    return ret;
}



/*
 * Calculates dot product between 1d base vector and 2d hyper plane
 * returns a unique hash identifying the base vector
*/
string dot(std::vector<double> base_vector,std::vector<std::vector<double>> plane )
{
    std::string str;
    int n = plane[0].size();
    int m = base_vector.size();
    std::vector<double> ret;
    ret.resize(n);
    
    for (int i = 0; i < n; i++)
    {
        ret[i] = 0;
        
        for (int j = 0; j < m; j++)
        { 
          ret[i] +=  base_vector[j] * plane[j][i]; 
        }
        ret[i] = (ret[i]>0)? 1 : 0 ;
        str.push_back((int)ret[i] + '0');
    }
    return str;
}


/*
 * Returns the hamming distance between two strings
 * aka the nubmer of differences
 */
int hammingDist(string str1, string str2)
{
    int i = 0, count = 0;
    while (str1[i] != '\0') 
    {
      if (str1[i] != str2[i]) {count++;}   
      i++;
    }
    return count;
}


int main(int argc, char* argv[]) 
{
  auto start = high_resolution_clock::now();
  FILE* fin = fopen(argv[1],"r");
  FILE* fout = fopen(argv[2],"w");
  std::ifstream file;
  file.open(argv[1]);
  int n = 0,d = 0,m = 0;
  double target_similarity = 0;
  fscanf(fin,"%d%d%d%lf",&d,&n,&m,&target_similarity);fclose(fin);
  printf("Reading file parameters...          ||  File Parameters: %d %d %d %lf  \n",d,n,m,target_similarity);
  

  // <<<<<<<<    Storing lines from txt file    >>>>>>>>
    printf("Reading vectors from txt file...  ");
    auto read_begin = high_resolution_clock::now();
    vector<string> lines = read_file(file);
    base = read_matrix(lines,n,d,1);
    query = read_matrix(lines,m,d,n+1);
    auto read_end = high_resolution_clock::now();
    auto read_duration = duration_cast<milliseconds>(read_end-read_begin);
    cout<<"  ||  Read Duration: "<<read_duration.count()*1.0/1000<<"s"<< endl;
  // <<<<<<<<    End of Storing lines from txt file    >>>>>>>>

  
  // <<<<<<<<<    Creating Hash table Process Starts Here    >>>>>>>>>
  printf("Creating Hash Tables...           ");
  auto hash_begin = high_resolution_clock::now();
  int nbits = ceil(sqrt(d));
  nbits = (nbits>12)? 13:nbits;
  std::vector<std::vector<double>> plane = hyperplane_generator(d,nbits);
  unordered_map<string, vector<int>> umap;
  #pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    string hash_str = dot(base[i],plane);
    #pragma omp critical
    {umap[hash_str].push_back(i);}  
  }
  auto hash_end = high_resolution_clock::now();
  auto hash_duration = duration_cast<milliseconds>(hash_end-hash_begin);
  cout<<"  ||  Hash Duration: "<<hash_duration.count()*1.0/1000<<"s\n";
  // <<<<<<<<<    End of Creating Hash table Process     >>>>>>>>>  
  
  
  printf("Starting Similarity Computations...\n");
  printf("Vector planes: %d    ",nbits);
  printf("Base vectors: %d    ",n);
  printf("Buckets: %d    \n",umap.size());
  int unknown = 0; // counts unsatisfied queries
  
  
  //<<<<<<<<<<    Similarity computations begin    >>>>>>>>>>
  for(int i=0;i<m;i++)
  {
   string q_hash = dot(query[i],plane); //obtaining query hash
   if(umap.find(q_hash) != umap.end()) 
    {
     int index = -1;
     for(int id:umap[q_hash])
     {
      double d = cosine_similarity(query[i],base[id]);
      if(d >= target_similarity)
      {
        index = id;
        break;
      }
     }
     unknown = (index != -1)? unknown : unknown+1; //uncomment to find number of unsatissfied queries
     fprintf(fout,"%d\n",index);
    }
    else 
    {
      string approx_hash;
      int min=INT_MAX;
      for(auto kv : umap) 
      {
       
        int dist = hammingDist(q_hash,kv.first);
        if(dist<min)
        {
          min=dist;
          approx_hash = kv.first;
        }
      } 

     int index = -1;
     for(int id:umap[approx_hash])
     {
      double d = cosine_similarity(query[i],base[id]);
      if(d >= target_similarity)
      {
        index = id;
        break;
      }
     }
     unknown = (index != -1)? unknown : unknown+1; //uncomment to find number of unsatissfied queries
     fprintf(fout,"%d\n",index);
    }
  }


  fclose(fout);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop-start);
  double time = (duration.count())*1.0/1000;
  cout<<"\n+--------------------------------------"<<endl;
  cout<<"|      Vector Similarity Computed   \n";
  cout<<"|--------------------------------------"<<endl;
  cout<<"| Total Duration     : "<<time<<"s"<< endl;
  cout<<"| Unsatisfied Queries: "<<unknown<<"/"<<m;
  cout<<"\n+--------------------------------------"<<endl;
  
  return 0;
}