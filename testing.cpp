//
//  testing.cpp
//  neural network
//
//  Created by Tao Jiang on 5/10/16.
//  Copyright © 2016 Tao Jiang. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace std;

int checkResult(vector<double>& temp);

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);
    
    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals, vector<double> &targetOutputVals);
    //    unsigned getTargetOutputs(vector<double> &targetOutputVals);
    
private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;
    
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }
    
    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    
    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals, vector<double> &targetOutputVals)
{
    inputVals.clear();
    targetOutputVals.clear();
    
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    
    //    string label;
    //    ss>> label;
    //    if (label.compare("in:") == 0) {
    for (int i = 0; i < 256; i++) {
        double oneValue;
        ss >> oneValue;
        inputVals.push_back(oneValue);
    }
    
    for (int i = 0; i < 10; i++){
        double oneValue;
        ss >> oneValue;
        targetOutputVals.push_back(oneValue);
    }
    //    }
    
    return inputVals.size();
}

//unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
//{
//    targetOutputVals.clear();
//
//    string line;
//    getline(m_trainingDataFile, line);
//    stringstream ss(line);
//
////    string label;
////    ss>> label;
////    if (label.compare("out:") == 0) {
//    for (int i = 0; i < 10; i++){
//        double oneValue;
//        while (ss >> oneValue) {
//            targetOutputVals.push_back(oneValue);
//        }
//    }
////    }
//
//    return targetOutputVals.size();
//}



struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight() {return rand() / double(RAND_MAX);}
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
    
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val;}
    double getOutputVal(void) const { return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};

double Neuron::eta = 0.15;   // overall net learning rate;
double Neuron::alpha = 0.5;

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned i = 0; i < numOutputs; i++) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    
    m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
        prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    
    // Sum our contributions of the errors at the nodes we feed.
    
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient +
        alpha * oldDeltaWeight;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

// ****************** class Net ******************

class Net
{
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
    
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const ;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    
    void showDetails();
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum < topology[layerNum] + 1; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            //            cout << "Made a Neuron!" << endl;
        }
        
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);
    
    for (unsigned i = 0; i < inputVals.size(); i++){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS
    
    // Implement a recent average measurement
    m_recentAverageError =
    (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
    / (m_recentAverageSmoothingFactor + 1.0);
    
    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    
    // Calculate hidden layer gradients
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    
    // For all layers from outputs to first hidden layer,
    // update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();
    
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::showDetails()
{
    for (unsigned layerNum = 0; layerNum < m_layers.size(); layerNum++) {
        cout<<"layerNum: "<<layerNum<<endl;
        for (unsigned n = 0; n < m_layers[layerNum].size(); n++) {
            cout<<"neural No: "<< n + 1 <<", value: "<<m_layers[layerNum][n].getOutputVal()<<"."<<endl;
        }
    }
}


void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    
    cout << endl;
}

void draw(vector<double> &v)
{
    for(int i = 0; i < 16; i++){
        for( int j = 0; j < 16; j++){
            if(v[i*16 + j] == 1.0){
                cout<<"X";
            } else{
                cout<<" ";
            }
        }
        cout<<endl;
    }
}

void targets(vector<double> &v){
    for (int i = 0; i < v.size(); i++) {
        if(v[i] == 1.0)
            cout<<"the result is: "<<i<<endl;
    }
}

//***********************product a trainning data*******************

//#include <iostream>
//#include <cmath>
//#include <cstdlib>
//#include <fstream>
//
//using namespace std;
//
//int main(){
//
//    ofstream myfile("/Users/Jhzhou/Desktop/neural network/fortest2.txt");
//    if(myfile.fail()){
//        cout<<"you can not open the file.";
//    }
//    myfile <<"topology: 1 4 3 1"<<endl;
//    for (int i = 15000; i >= 0; i—) {
//        double n1 = (rand() / double(RAND_MAX)) * 6.4 - 3.2;
//        double t = sin(n1);
//        myfile<<"in: "<< n1 << " "<<endl;
//        myfile<<"out: "<< t << " "<<endl;
//    }
//}

//***********************test for a trainning data*******************
//
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <sstream>
//
//using namespace std;
//void showVectorVals(string label, vector<double> &v);
//
//int main()
//{
//
//    ifstream myfile("/Users/Jhzhou/Desktop/fortest3.txt");
//    if(myfile.fail()){
//        cout<<"you can not open the file."<<endl;
//    }
//    string line;
//    getline(myfile, line);
//    vector<double> inputValues;
//    for (int i = 0; i < line.size(); i++) {
//        if (isdigit(line[i])) {
//            inputValues.push_back(line[i] - '0');
//        }
//    }
//    showVectorVals("input: ", inputValues);
//    return 0;
//
//}

//*************************copy main function************************

int main()
{
    TrainingData trainData("/Users/Jhzhou/Desktop/neural network/forTest2.txt");
    
    vector<unsigned> topology;
    trainData.getTopology(topology);
    
    Net myNet(topology);
    
    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    
    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;
        
        if (trainData.getNextInputs(inputVals, targetVals) != topology[0]) {
            cout <<"lalla"<<endl;
            cout<< topology[0]<<endl;
            cout<<inputVals.size()<<endl;
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        draw(inputVals);
        myNet.feedForward(inputVals);
        
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);
        
        // Train the net what the outputs should have been:
        //        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        targets(targetVals);
        assert(targetVals.size() == topology.back());
        
        myNet.backProp(targetVals);
        
        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
        << myNet.getRecentAverageError() << endl;
    }
    
    //    int testPass = 0;
    //    while(!trainData.isEof()){
    //        testPass++;
    //        cout << endl << "TestPass " << testPass;
    //        if (trainData.getNextInputs(inputVals, targetVals) != topology[0]) {
    //            cout <<"lalla"<<endl;
    //            cout<< topology[0]<<endl;
    //            cout<<inputVals.size()<<endl;
    //            break;
    //        }
    //        showVectorVals(": Inputs:", inputVals);
    //        draw(inputVals);
    //        myNet.feedForward(inputVals);
    //
    //        myNet.getResults(resultVals);
    //        showVectorVals("Outputs:", resultVals);
    //
    //        cout << "Net recent average error: "
    //        << myNet.getRecentAverageError() << endl;
    //    }
    
    ifstream myfile("/Users/Jhzhou/Desktop/forTest3.txt");
    if(myfile.fail()){
        cout<<"you can not open the file."<<endl;
    }
    string line;
    getline(myfile, line);
    stringstream ss(line);
    vector<double> inputValues2;
    
    for (int i = 0; i < line.size(); i++) {
        if (isdigit(line[i])) {
            inputValues2.push_back((line[i]-'0')/1.0);
        }
    }
    showVectorVals("input of jiahuang zhou: ", inputValues2);
    draw(inputValues2);
    myNet.feedForward(inputValues2);
    myNet.getResults(resultVals);
    
    showVectorVals("jiahuang zhou's test: ", resultVals);
    myNet.showDetails();
    
    return 0;
}
