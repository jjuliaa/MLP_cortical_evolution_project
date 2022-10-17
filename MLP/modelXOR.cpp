#include <iostream>
#include <vector>
#include <math.h>
#include "MLP.h"
using namespace std;


int main(){
    

    int T=100000;
    
    MLP M(2,3, 3,1);
    
    
    //XOR problem
    vector<vector<double> > input(4, vector<double> (2, 0.0));
    input[1][1] = 1.0;
    input[2][0] = 1.0;
    input[3][0] = 1.0;
    input[3][1] = 1.0;
    
    vector<double> target(4, 0.0);
    target[1] = 1.0;
    target[2] = 1.0;
    
    
    for (int t = 0; t<T; t++){
       int r = (int)randDbl(0.0, 4.0);
        M.update(0.05, input[r], vector<double> (1, target[r]), vector<int> (0)); //no element (0)
       }
    
    for (int i=0; i<input.size(); i++){
        M.update(0.0, input[i], vector<double> (M.nK, 0.0), vector<int> (0));
        
        std::cout<<"Output "<<M.Ks[0]<<std::endl;
    }
    
    
    
    return 0;
   
}





