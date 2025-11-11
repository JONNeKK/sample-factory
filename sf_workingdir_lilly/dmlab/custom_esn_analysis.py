import torch
from torch import nn 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from sf_workingdir_lilly.dmlab.custom_weight_generator import return_weights_for_spec_rad


def get_hidden_states(time_steps, spectral_radius, Hippo_n_feature, Hippo_R, Hippo_L, trial=1):

    # create the rnn
    W_ih, W_hh = return_weights_for_spec_rad(spectral_radius, trial)

    expanded_length = Hippo_R + Hippo_L - 1
    hidden_size = Hippo_n_feature * expanded_length

    rnn = nn.RNN(input_size=Hippo_n_feature, 
                          hidden_size=hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False,
                          bias=False)
    
    # fix the rnn weights
    W_ih = W_ih.detach()
    W_hh = W_hh.detach()
    W_ih = W_ih.to(rnn.weight_ih_l0.device, dtype=rnn.weight_ih_l0.dtype)
    W_hh = W_hh.to(rnn.weight_hh_l0.device, dtype=rnn.weight_hh_l0.dtype)
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(W_ih)
        rnn.weight_hh_l0.copy_(W_hh)
    # Freeze RNN parameters.
    for param in rnn.parameters():
        param.requires_grad = False
    # create output vector (time steps x hidden size)
    output = torch.zeros(time_steps, hidden_size)

    # first input
    u = torch.zeros(1,1,Hippo_n_feature) # shape: (1, B, n_feature)
    u[:,0,0] = 1.0
    h0 = torch.zeros(1,1,hidden_size)
    y, h = rnn.forward(u, h0)
    output[0,:] = h

    # zero inputs for the amount of time steps
    for i in range(1, time_steps):
        u = torch.zeros(1,1,Hippo_n_feature)
        y, h = rnn.forward(u, h)
        # update the output vector: update the row corresponding to the current time step
        output[i,:] = h
        i += 1

    # return output vector
    return output


def esn_pca(hidden_states):
    hs_np = hidden_states.detach().numpy()
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(hs_np)
    return transformed_data

def visualize_pca_results(transformed_data, path='pca_hidden.png'):
    plt.figure()
    plt.plot(transformed_data[:,0],transformed_data[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()


def main():
    spectral_radius = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99]

    hs = get_hidden_states(70, 0.5, 16, 8, 64)  
    print(hs)
    td = esn_pca(hs)
    print(td)

    # visualize_pca_results(td)

if __name__ == "__main__":
    main()

