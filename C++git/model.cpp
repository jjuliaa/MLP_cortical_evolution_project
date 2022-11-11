#include <iostream>
#include <vector>
#include <math.h>
#include "MLP.h"
#include <morph/HdfData.h>
#include <morph/Config.h>
#include "morph/vVector.h"

using namespace std;

std::vector<double> ScaleX(vector<double> x1, vector<double> y1) {
    std::vector<double> x1_s = x1;
    int maxx=0;
    for(int i=0;i<x1.size();i++){
        if(x1[i]>maxx){
            maxx=x1[i];
        }
    }
    int maxy= 0;
    for(int i=0;i<y1.size();i++){
        if(y1[i]>maxy){
            maxy=y1[i];
        }
    }
    int maxDim = maxx;
    if(maxy> maxx){
        maxDim = maxy;
    }
    for(int i=0;i<x1.size();i++){
        x1_s[i] = x1[i]/(double)maxDim;
    }
    return x1_s;
}

std::vector<double> ScaleY(vector<double> x1, vector<double> y1) {
    std::vector<double> y1_s = y1;
    
    int maxx=0;
    for(int i=0;i<x1.size();i++){
        if(x1[i]>maxx){
            maxx=x1[i];
        }
    }
    int maxy= 0;
    for(int i=0;i<y1.size();i++){
        if(y1[i]>maxy){
            maxy=y1[i];
        }
    }
    int maxDim = maxx;
    if(maxy> maxx){
        maxDim = maxy;
    }
    for(int i=0;i<y1.size();i++){
        y1_s[i] = y1[i]/(double)maxDim;
    }
    return y1_s;
}


int main(int argc, char **argv){
    std::vector<double> x1;
    std::vector<double> y1;
    std::vector<double> c1;
    morph::HdfData data("img1coded.h5", morph::FileAccess::ReadOnly);
    
    std::vector<double> x2;
    std::vector<double> y2;
    std::vector<double> c2;
    morph::HdfData data2("img2coded.h5", morph::FileAccess::ReadOnly);
    
    data.read_contained_vals ("x1", x1);
    std::cout<<"x1"<<std::endl;
    data.read_contained_vals ("y1", y1);
    std::cout<<"y1"<<std::endl;
    data.read_contained_vals ("c1", c1);
    std::cout<<"c1"<<std::endl;
    
    data2.read_contained_vals ("x2", x2);
    std::cout<<"x2"<<std::endl;
    data2.read_contained_vals ("y2", y2);
    std::cout<<"y2"<<std::endl;
    data2.read_contained_vals ("c2", c2);
    std::cout<<"c2"<<std::endl;
    
    //scale inputs
    std::vector<double> x1_s = ScaleX(x1, y1);
    std::vector<double> y1_s = ScaleY(x1, y1);
    std::vector<double> x2_s = ScaleX(x2, y2);
    std::vector<double> y2_s = ScaleY(x2, y2);
    
    // read from a config file (json)
    std::string paramsfile (argv[1]);
    morph::Config conf(paramsfile);
    if (!conf.ready) { std::cerr << "Error setting up JSON config: " << conf.emsg << std::endl; return 1; }
    
    // create log file
    std::string logpath = argv[2];
    std::ofstream logfile;
    morph::Tools::createDir (logpath);
    { std::stringstream ss; ss << logpath << "/log.txt"; logfile.open(ss.str());}
    logfile<<"Hello World."<<std::endl;
    
    int T = conf.getInt("T", 1000);
    int nHiddenJ = conf.getInt("nHiddenJ", 3);
    int nHiddenQ = conf.getInt("nHiddenQ", 3);
    //int learningRate = conf.getFloat("learningRate", 0.03);
    //to run: ./model ../config.json logs
    
    MLP M(2,nHiddenJ,nHiddenQ,3);
    
    //Image Training:
    for (int t = 0; t<T; t++){
        
        vector<double> input(2, 0.0);
        vector<double> target(3, 0.0);
        int r = (int)randDbl(0.0, x1.size());
        //Q: how to pass directly into funct?
        vector<int> firstnode(1, 0);
        vector<int> secondnode(1, 1);
        
        float context = (float)randDbl(0,1)<0.5;
        int areaFlag = 0; //the color codes
        
         /*input[0] = x1_s[r];
         input[1] = y1_s[r];
         input2[0] = x2_s[r];
         input2[1] = y2_s[r];
         */
        
        //context nodes
        if(context){
            r = (int)randDbl(0.0, x1.size()); //Q: does this need to be the np.floor of?
            input[0] = x1_s[r];
            input[1] = y1_s[r];
            areaFlag = c1[r]; //Q: does c1 also need scaling?!
        } else {
            r = (int)randDbl(0.0, x2.size());
            input[0] = x2_s[r];
            input[1] = y2_s[r];
            areaFlag = c2[r];
        }
        
        //set targets (see python if needs redo)
        if (areaFlag > 0.0) {
            target[areaFlag-1] = 1.0; //draw a random x, y color
        }
        
        if((t%100)==0){
            std::cout << "progress: " <<(float)t/T << std::endl;
        }
        
        if(context){
            M.update(0.07, input, target, firstnode); //sending 0 to disable the node at index 0, for no context nodes vector<int> (0) is no element (0)
        } else {
            M.update(0.07, input, target, secondnode);
        }
    }
    
    
    std::cout<<"Finished Training"<<std::endl;
    
    
    // Testing map 1
    vector<double> outNode1(0.0);
    vector<double> outNode2(0.0);
    vector<double> outNode3(0.0);
    
    for (int r = 0; r<x1.size(); r++){
        vector<double> input(2, 0.0);
        vector<double> target(3, 0.0);
        
        input[0] = x1_s[r];
        input[1] = y1_s[r];
        vector<int> firstnode(1, 0);
        M.update(0.00, input, target, firstnode);
        outNode1.push_back(M.Ks[0]);
        outNode2.push_back(M.Ks[1]);
        outNode3.push_back(M.Ks[2]);
        
    }
    
    //Testing map 2
    vector<double> outNodeA1(0.0);
    vector<double> outNodeA2(0.0);
    vector<double> outNodeA3(0.0);
    
    
    for (int r = 0; r<x2.size(); r++){
        vector<double> input(2, 0.0);
        vector<double> target(3, 0.0);
        
        input[0] = x2_s[r];
        input[1] = y2_s[r];

        vector<int> secondnode(1, 1);
        M.update(0.00, input, target, secondnode);
        outNodeA1.push_back(M.Ks[0]);
        outNodeA2.push_back(M.Ks[1]);
        outNodeA3.push_back(M.Ks[2]);
        
    }
    
    //Interpolate map 3 (1 and 2 ancestor)
    vector<double> outNodeB1(0.0);
    vector<double> outNodeB2(0.0);
    vector<double> outNodeB3(0.0);
    
    
    for (int r = 0; r<x2.size(); r++){
        vector<double> input(2, 0.0);
        vector<double> target(3, 0.0);
        
        //for interp map 3 only change here A
        input[0] = x1_s[r]; //how do I use both as inputs here?
        input[1] = y1_s[r];
        //input[2] = x2_s[r];
        //input[3] = y2_s[r];
        
        M.update(0.00, input, target, vector<int> (0) ); //no element (0) all nodes enabled, ancestor
        outNodeB1.push_back(M.Ks[0]);
        outNodeB2.push_back(M.Ks[1]);
        outNodeB3.push_back(M.Ks[2]);
        
    }
    
    //Interpolate map 4 (1 and 2 progeny)
     vector<double> outNodeC1(0.0);
     vector<double> outNodeC2(0.0);
     vector<double> outNodeC3(0.0);
     
     
     for (int r = 0; r<x2.size(); r++){
         vector<double> input(2, 0.0);
         vector<double> target(3, 0.0);
         
         vector<int> bothNodes;
         bothNodes.push_back(0);
         bothNodes.push_back(1);
         
         //for interp map only change here B
         input[0] = x1_s[r]; //how do I use both as inputs here?
         input[1] = y1_s[r];
         //input[2] = x2_s[r];
         //input[3] = y2_s[r];
         
         M.update(0.00, input, target, bothNodes); //no element (0) all nodes enabled, ancestor
         outNodeC1.push_back(M.Ks[0]);
         outNodeC2.push_back(M.Ks[1]);
         outNodeC3.push_back(M.Ks[2]);
         
     }
    
    std::cout<<"Finished Testing..."<< std::endl;
    //Save output errors
    {
        std::stringstream fname;
        fname << logpath << "/out.h5";
        morph::HdfData dataout(fname.str());
        std::stringstream path;
        
        path.str("");path.clear();path << "/errors";
        dataout.add_contained_vals (path.str().c_str(), M.Errors);
        
        path.str("");path.clear();path << "/output1";
        dataout.add_contained_vals (path.str().c_str(), outNode1);
        
        path.str("");path.clear();path << "/output2";
        dataout.add_contained_vals (path.str().c_str(), outNode2);
        
        path.str("");path.clear();path << "/output3";
        dataout.add_contained_vals (path.str().c_str(), outNode3);
        
        path.str("");path.clear();path << "/x1_s";
        dataout.add_contained_vals (path.str().c_str(), x1_s);
        
        path.str("");path.clear();path << "/y1_s";
        dataout.add_contained_vals (path.str().c_str(), y1_s);
        
        path.str("");path.clear();path << "/x1";
        dataout.add_contained_vals (path.str().c_str(), x1);
        
        path.str("");path.clear();path << "/y1";
        dataout.add_contained_vals (path.str().c_str(), y1);
        
        // second map
        path.str("");path.clear();path << "/outputA1";
        dataout.add_contained_vals (path.str().c_str(), outNodeA1);
        
        path.str("");path.clear();path << "/outputA2";
        dataout.add_contained_vals (path.str().c_str(), outNodeA2);
        
        path.str("");path.clear();path << "/outputA3";
        dataout.add_contained_vals (path.str().c_str(), outNodeA3);
        
        path.str("");path.clear();path << "/x2_s";
        dataout.add_contained_vals (path.str().c_str(), x2_s);
        
        path.str("");path.clear();path << "/y2_s";
        dataout.add_contained_vals (path.str().c_str(), y2_s);
        
        path.str("");path.clear();path << "/x2";
        dataout.add_contained_vals (path.str().c_str(), x2);
        
        path.str("");path.clear();path << "/y2";
        dataout.add_contained_vals (path.str().c_str(), y2);
        
        //interpolation map ancestor
        path.str("");path.clear();path << "/outputB1";
        dataout.add_contained_vals (path.str().c_str(), outNodeB1);
        
        path.str("");path.clear();path << "/outputB2";
        dataout.add_contained_vals (path.str().c_str(), outNodeB2);
        
        path.str("");path.clear();path << "/outputB3";
        dataout.add_contained_vals (path.str().c_str(), outNodeB3);
        
        //interpolation map progeny
          path.str("");path.clear();path << "/outputC1";
          dataout.add_contained_vals (path.str().c_str(), outNodeC1);
          
          path.str("");path.clear();path << "/outputC2";
          dataout.add_contained_vals (path.str().c_str(), outNodeC2);
          
          path.str("");path.clear();path << "/outputC3";
          dataout.add_contained_vals (path.str().c_str(), outNodeC3);
        

    }
    
    
    return 0;
    
}
