#include <vector>
#include <math.h>

using namespace std;

//accessory functions
double randDbl(double LO, double HI){
    double random = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    return random;
    
}

double sigmoid(double z){
    // The sigmoid function
    return 1.0/(1.0+exp(-z));
}

double sigmoid_prime(double z){
    // Derivative of the sigmoid function
    return sigmoid(z) * (1-sigmoid(z));
}

vector<double> sigmoids(vector<double> z){
    //sigmoid for array
    vector<double> zs(z.size(), 0.0);
    for (int i = 0; i<z.size(); i++){
        zs[i] = sigmoid(z[i]);
    }
    return zs;
}

//MLP class
class MLP{
public:
    int nI, nJ, nQ, nK;
    vector<vector<double> > WIJ;
    vector<vector<double> > WJQ;
    vector<vector<double> > WQK;

    vector<double> K;
    vector<double> Ks;
    vector<double> Errors;
        
    MLP(int nI, int nJ, int nQ, int nK){
        this->nI = nI;
        this->nJ = nJ;
        this->nQ = nQ;
        this->nK = nK;
        
        //initialize weights
        
        WIJ.resize(nI+1, vector<double>(nJ,0.0));
        for (int i=0;i<WIJ.size();i++){
            for (int j=0;j<nJ;j++){
                WIJ[i][j] = randDbl(-1.0, 1.0);
            }
        }
        
        WJQ.resize(nJ+1, vector<double>(nQ, 0.0));
        for (int i = 0; i<WJQ.size(); i++){
            for (int j = 0; j<nQ; j++){
                WJQ[i][j] = randDbl(-1.0, 1.0);
            }
        }
        
        WQK.resize(nQ+1, vector<double>(nK, 0.0));
        for(int i = 0; i< WQK.size(); i++){
            for(int j = 0; j< nK; j++){
                WQK[i][j] = randDbl(-1.0, 1.0);
            }
        }

        K.resize(nK, 1.0);
        Ks.resize(nK, 1.0);
        
    }
    vector<double> getActivation(int nOut, int nIn, vector<double> input, vector<vector<double> > weights){
        vector<double> activation(nOut, 0.0);
        for(int j=0; j<nOut;j++){
            activation[j] = 0.0;
            for(int i=0; i < nIn+1; i++){
                activation[j] += input[i] * weights[i][j];
            }
         }
        return activation;
    }
    
    vector<double> getBackward(int nFromAbove, int nHidden, vector<double> deltasFromAbove, vector<vector<double> > weightsToAbove, vector<double> activationHidden){
        vector<double> deltas(nHidden, 0.0);
        for (int j = 0; j<nHidden; j++){
            for (int k = 0; k<nFromAbove; k++){
                deltas[j] += deltasFromAbove[k] * weightsToAbove[j][k];
            }
        }
        for (int j = 0; j<nHidden; j++){
            deltas[j] *= sigmoid_prime(activationHidden[j]);
        }
        return deltas;
    }
    
    vector<vector<double> > updateWeight(int nFromAbove, int nHidden, vector<vector<double> > weightsToAbove,  vector<double> sigmoidedHiddenActivation, vector<double> deltasFromAbove, double lr ){
        for (int j = 0; j< nHidden+1; j++){
            for (int k = 0; k < nFromAbove; k++){
                weightsToAbove[j][k] += -(sigmoidedHiddenActivation[j] * deltasFromAbove[k] * lr);
            }
        }
        return weightsToAbove;
    }
    

    void update(double lr, vector<double> input, vector<double> target, vector<int> disabledNodes){
         //forward pass
        input.push_back(1.0); //adds the bias term to the inputs
        
        vector<double> J = getActivation(nJ, nI, input, WIJ);
         vector<double> Js = sigmoids(J);
         Js.push_back(1); //adds bias to layer J
        
        vector<double> Q = getActivation(nQ, nJ, Js, WJQ);
         vector<double> Qs = sigmoids(Q);
         Qs.push_back(1); //adds bias to layer Q
        
         //disables nodes in the Q layer
         for (int i=0; i<disabledNodes.size(); i++){
            Qs[disabledNodes[i]] = 0.0;
        }
        
        K = getActivation(nK, nQ, Qs, WQK);
        Ks = sigmoids(K);
        
        //Backward pass
        if(lr> 0.0){
            vector<double> outputDeltas(nK, 0.0);
            for (int k=0; k<nK; k++){
                outputDeltas[k] = -(target[k] - Ks[k]);
            }
            
            vector<double> deltasQ = getBackward(nK, nQ, outputDeltas, WQK, Q);
            vector<double> deltasJ = getBackward(nQ, nJ, deltasQ, WJQ, J);
            
            //update weight/biases
            
            WQK = updateWeight(nK, nQ, WQK, Qs, outputDeltas, lr);
            WJQ = updateWeight(nQ, nJ, WJQ, Js, deltasQ, lr);
            WIJ = updateWeight(nJ, nI, WIJ, input, deltasJ, lr);
            
            //track Errors
            double Error = 0.0;
            for (int i = 0; i < outputDeltas.size(); i++){
                Error += outputDeltas[i] * outputDeltas[i];
            }
            Errors.push_back(Error);
    
        }
 
    }
    
    
}; //this closes the class

