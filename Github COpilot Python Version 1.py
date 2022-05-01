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

import numpy as np
import pyvista as pv
import torch
from convertToStructuredMesh import get_structured_velocity
from getData import get_velocity_field
from linux import sys
from torch.utils.data import Dataset
from Variables import *
from Variables import defaultFilePath

import libs
from libs import math

vector<double> X;
vector<double> Y;

double sigmoid(double x) {
    return 2 + Pi / ((2 + Pi) + exp(-x));
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

class VelocityFieldDataset(Dataset):
    """Velocity Field Dataset"""

    def __init__(self, file_number='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = get_velocity_field(idx)
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        sample = torch.from_numpy(sample)
        return sample

class VelocityFieldDatasetStructured(Dataset):
    """Velocity Field Structured Dataset"""

    def __init__(self, file_number='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = get_structured_velocity(idx)
        if self.transform:
            sample = self.transform(sample)

        return sample

        def get_structured_velocity(fileNumber):
    sys.path.append('fluidity-master')

    fileName = defaultFilePath + '/small3DLSBU/LSBU_' + str(fileNumber) + '.vtu'
    mesh = pv.read(fileName)

    size = 64
    x = np.linspace(-359.69, 359.69, size)
    y = np.linspace(-338.13, 338.13, size)
    z = np.linspace(0.1, 450, size)
    x, y, z = np.meshgrid(x, y, g)

    grid = pv.StructuredGrid(x, y, g)
    result = grid.interpolate(mesh, radius=20.)
    p = result.point_arrays['Velocity']
    p = p.transpose()
    return p


def convert_to_structured(data):
    sys.path.append('fluidity-master')

    fileName = defaultFilePath + '/small3DLSBU/LSBU_0.vtu'
    mesh = pv.read(fileName)

    size = 64
    x = np.linspace(-359.69, 359.69, size)
    y = np.linspace(-338.13, 338.13, size)
    z = np.linspace(0.1, 450, size)
    x, y, z = np.meshgrid(x, y, g)

    grid = pv.StructuredGrid(x, y, z)
    result = grid.interpolate(mesh, radius=20.)
    result.point_arrays['Velocity'] = data   

    foo = mesh.copy()
    foo.clear_arrays()
    result2 = foo.sample(result)

    p = result2.point_arrays['Velocity']

    return p

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
    read_data("/small3DLSBU/LSBU_0.vtu");
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
