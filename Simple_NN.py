import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def mse_loss(y_true,y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    
    # A neural network with:
    #   - 4 inputs
    #   - a hidden layer with 4 neurons
    #   - an output layer with 1 neuron
    
    def __init__(self):
        self.weights_hidden = np.random.normal(size=(4,4))
        self.bias_hidden = np.random.normal(size=(4,))
        self.weights_output = np.random.normal(size=(4,1))
        self.bias_output = np.random.normal(size=(1,))
    
    def feedforward(self, data):
        hidden_output = sigmoid((np.dot(data, self.weights_hidden) + self.bias_hidden))
        output = sigmoid(np.dot(hidden_output, self.weights_output) + self.bias_output)
        return output
            
    def train(self, data, all_trues, epochs, learning_rate):
        
        low_loss = 1
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(data, all_trues):
        
                #obtain numbers
                hidden_outputs = np.dot(x, self.weights_hidden) + self.bias_hidden
                hidden_outputs_activated = sigmoid(hidden_outputs)
                output = np.dot(hidden_outputs_activated, self.weights_output) + self.bias_output
                
                output_activated = sigmoid(output)              
                
                #loss calculation
                y_pred = output_activated
                loss = mse_loss(np.array([y_true]), np.array([y_pred]))
                total_loss += loss
    
                dl_dpred = -(y_true - y_pred)
                
                #gradient for output layer
                d_out = dl_dpred * deriv_sigmoid(output)
              
                partial_derivatives_output_weights = hidden_outputs * d_out
                partial_derivatives_output_bias =  d_out
                
                #gradient for hidden layer 
                
                d_out_hidden = d_out * deriv_sigmoid(hidden_outputs).reshape(-1,1) * self.weights_output
            
                partial_derivatives_hidden_weights = x.reshape(-1,1) @ d_out_hidden.reshape(1,-1)
                partial_derivatives_hidden_bias =  d_out_hidden
                
                #change weights and biases
                
                self.weights_hidden -= learning_rate * partial_derivatives_hidden_weights
                self.bias_hidden -= learning_rate * partial_derivatives_hidden_bias.reshape(-1)
                self.weights_output -= learning_rate * partial_derivatives_output_weights.reshape(-1,1)
                self.bias_output -= learning_rate * partial_derivatives_output_bias
                
            if epoch % 50 == 0:
                avg_loss = total_loss / len(data)
                if avg_loss < low_loss:
                    low_loss = avg_loss
                elif epoch > 500:
                    break
                print(f"Epoch {epoch} loss: {avg_loss:.3f}")
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)