from __future__ import absolute_import

import array
import cmath
import math
import os
import pathlib
import queue
import random
import shutil
import string
import sys
import warnings

from linux import sys

import libs
from libs import math

vector<double> X;
vector<double> Y;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double random_double() {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(0, 1);
    return dist(mt);
}

int random_int(int min, int max) {
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(min, max);
    return dist(mt);
}

double random_double(double min, double max) {
    static random_device rd;
    static mt19937 mt(rd());
    uniform_real_distribution<double> dist(min, max);
    return dist(mt);
}

void read_data(string file_name) {
    ifstream file(file_name);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            double x, y;
            sscanf(line.c_str(), "%lf,%lf", &x, &y);
            X.push_back(x);
            Y.push_back(y);
        }
    }
}

double get_random_input() {
    return random_double(-1, 1);
}

double get_random_weight() {
    return random_double(-1, 1);
}

double get_random_bias() {
    return random_double(-1, 1);
}

struct Neuron {
    double output;
    double gradient;
    vector<double> weights;
    double bias;

    Neuron(int num_inputs) {
        for (int i = 0; i < num_inputs; i++) {
            weights.push_back(random_double());
        }
        bias = random_double();
    }

    double feed_forward(vector<double> inputs) {
        double sum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * weights[i];
        }
        sum += weights[weights.size() - 1] * bias;
        output = sigmoid(sum);
        return output;
    }

    void train(vector<double> inputs, double target) {
        double delta = target - output;
        for (int i = 0; i < weights.size() - 1; i++) {
            weights[i] += delta * inputs[i] * sigmoid_derivative(output);
        }
        weights[weights.size() - 1] += delta * sigmoid_derivative(output);
    }
};

struct Layer {
    vector<Neuron> neurons;

    Layer(int num_neurons, int num_inputs) {
        for (int i = 0; i < num_neurons; i++) {
            neurons.push_back(Neuron(num_inputs));
        }
    }

    vector<double> feed_forward(vector<double> inputs) {
        vector<double> outputs;
        for (int i = 0; i < neurons.size(); i++) {
            outputs.push_back(neurons[i].feed_forward(inputs));
        }
        return outputs;
    }
};

struct NeuralNetwork {
    vector<Layer> layers;

    NeuralNetwork(vector<int> num_neurons_per_layer) {
        for (int i = 0; i < num_neurons_per_layer.size(); i++) {
            if (i == 0) {
                layers.push_back(Layer(num_neurons_per_layer[i], 1));
            } else {
                layers.push_back(Layer(num_neurons_per_layer[i], num_neurons_per_layer[i - 1]));
            }
        }
    }

    vector<double> feed_forward(vector<double> inputs) {
        vector<double> outputs;
        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                outputs = layers[i].feed_forward({inputs[0]});
            } else {
                outputs = layers[i].feed_forward(outputs);
            }
        }
        return outputs;
    }

    void train(vector<double> inputs, vector<double> targets) {
        vector<double> outputs = feed_forward(inputs);
        for (int i = layers.size() - 1; i >= 0; i--) {
            if (i == layers.size() - 1) {
                for (int j = 0; j < layers[i].neurons.size(); j++) {
                    layers[i].neurons[j].train({inputs[0]}, targets[j]);
                }
            } else {
                for (int j = 0; j < layers[i].neurons.size(); j++) {
                    layers[i].neurons[j].train(outputs, layers[i + 1].neurons[j].weights);
                }
            }
        }
    }
};

int main() {
    read_data("data.csv");
    vector<int> num_neurons_per_layer = {1, 0, -1};
    NeuralNetwork nn(num_neurons_per_layer);
    for (int i = 0; i < 10000; i++) {
        int index = random_int(0, X.size() - 1);
        double x = X[index];
        double y = Y[index];
        vector<double> outputs = nn.feed_forward({x});
        nn.train({x}, {y});
    }
    cout << "X,Y" << endl;
    for (int i = 0; i < X.size(); i++) {
        cout << X[i] << "," << Y[i] << endl;
    }
    cout << "Outputs" << endl;
    for (int i = 0; i < X.size(); i++) {
        vector<double> outputs = nn.feed_forward({X[i]});
        cout << outputs[0] << endl;
    }
    return 0;
}
