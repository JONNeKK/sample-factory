import torch
from torch import Tensor, nn
from pathlib import Path 


def generate_weights_pytorch_esn(hidden_size, n_feature, sparsity, spectral_radius):
    """
    generate the input and recurrent weights for a single input feature

    based on the pytorch-esn library
    
    :param hidden_size: network size
    :param n_feature: number of features there are (has to correspond to the hidden_size)
    :param sparsity: sparsity of the recurrent connections in the network
    :param spectral_radius: spectral radius of the recurrent weights
    """
    w_ih = torch.Tensor(hidden_size, n_feature)
    w_ih.uniform_(-1, 1)
    w_hh = torch.Tensor(hidden_size * hidden_size)
    #print(w_hh.size())
    w_hh.uniform_(-1, 1)
    # add sparcity to the recurrent matrix
    if sparsity<1:
        zero_weights = torch.randperm(int(hidden_size * hidden_size))
        zero_weights = zero_weights[:int(hidden_size * hidden_size * (1 - sparsity))]
        w_hh[zero_weights] = 0
    # reshape & scale to the desired spectral radius
    w_hh = w_hh.view(hidden_size, hidden_size)
    abs_eigs = torch.abs(torch.linalg.eigvals(w_hh))
    w_hh = w_hh * (spectral_radius / torch.max(abs_eigs))
    return w_ih, w_hh

def generate_weights_isolated_features(expanded_length, hidden_size, n_feature, sparsity, spectral_radius):
    """
    generate the input and recurrent weight matrices for the ESN
    each input feature has different input and recurrent weights
    
    :param expanded_length: size of 'small' network for each feature
    :param hidden_size: network size
    :param n_feature: number of features
    :param sparsity: sparsity of the recurrent connections in the network
    :param spectral_radius: spectral radius of the recurrent weights
    """
    w_ih = torch.zeros(hidden_size, n_feature) # input weights (input, hidden)
    w_hh = torch.zeros(hidden_size, hidden_size) # recurrent weights (hidden, hidden)

    # fill the weight matrices with weights, only in the positions that correpsond to the correct feature
    for i in range(n_feature):
         wih_current, whh_current = generate_weights_pytorch_esn(expanded_length, 1, sparsity, spectral_radius)
         w_ih[i*expanded_length:i*expanded_length+expanded_length, i] = wih_current.squeeze(-1)
         w_hh[i*expanded_length: i*expanded_length+expanded_length, i*expanded_length: i*expanded_length+expanded_length] = whh_current

    return w_ih, w_hh

def generate_weights_isolated_features_fixed_whh(expanded_length, hidden_size, n_feature, sparsity, spectral_radius):
    """
    generate the input and recurrent weight matrices for the ESN
    each input feature has different input weights, but all input features have the same recurrent weights
    
    :param expanded_length: size of 'small' network for each feature
    :param hidden_size: network size
    :param n_feature: number of features
    :param sparsity: sparsity of the recurrent connections in the network
    :param spectral_radius: spectral radius of the recurrent weights
    """
    w_ih = torch.zeros(hidden_size, n_feature)
    w_hh = torch.zeros(hidden_size, hidden_size)

    # fill the weight matrices with weights, only in the positions that correpsond to the correct feature
    wih_current, whh = generate_weights_pytorch_esn(expanded_length, 1, sparsity, spectral_radius)
    for i in range(n_feature):
         w_ih[i*expanded_length:i*expanded_length+expanded_length, i] = wih_current.squeeze(-1)
         w_hh[i*expanded_length: i*expanded_length+expanded_length, i*expanded_length: i*expanded_length+expanded_length] = whh
         wih_current, whh_current = generate_weights_pytorch_esn(expanded_length, 1, sparsity, spectral_radius)

    return w_ih, w_hh

def generate_weights_isolated_features_fixed_whh_wih(expanded_length, hidden_size, n_feature, sparsity, spectral_radius):
    """
    generate the input and recurrent weight matrices for the ESN
    all features have the same recurrent and input weights
    
    :param expanded_length: size of 'small' network for each feature
    :param hidden_size: network size
    :param n_feature: number of features
    :param sparsity: sparsity of the recurrent connections in the network
    :param spectral_radius: spectral radius of the recurrent weights
    """
    w_ih = torch.zeros(hidden_size, n_feature)
    w_hh = torch.zeros(hidden_size, hidden_size)

    # fill the weight matrices with weights, only in the positions that correpsond to the correct feature
    wih_current, whh_current = generate_weights_pytorch_esn(expanded_length, 1, sparsity, spectral_radius)
    for i in range(n_feature):
         w_ih[i*expanded_length:i*expanded_length+expanded_length, i] = wih_current.squeeze(-1)
         w_hh[i*expanded_length: i*expanded_length+expanded_length, i*expanded_length: i*expanded_length+expanded_length] = whh_current

    return w_ih, w_hh


####################################


def return_weights_for_iso_feat(spectral_radius, trial=1, fixed_whh=False, fixed_wih=False, vary_sparsity=False, sparsity=0.2, folder='weights'):
    """
    returns the pre-generated weights for a specific weight trial
    mainly the weights in the folder weights were used (here there were 50 different weights generated aka 50 different weight trials)
    
    :param spectral_radius: spectral radius of recurrent weights
    :param trial: weight trial 
    :param fixed_whh: True if the recurrent weights are the same for all features
    :param fixed_wih: True if the input weights are the same for all features
    :param vary_sparsity: weights with different sparsities 
    :param sparsity: sparsity of the recurrent weights
    :param folder: folder in which the weights are saved
    """
    if fixed_whh and not fixed_wih:
        if vary_sparsity:
            path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights_iso_feat_fixed_whh{trial}_sparsity{sparsity}")
        else:
            path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights_iso_feat_fixed_whh{trial}")
    elif fixed_whh and fixed_wih:
        path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/{folder}/custom_weights_iso_feat_fixed_whh_wih{trial}")
    else:
        path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights_iso_feat{trial}")
    
    path.mkdir(parents=True, exist_ok=True)
    file_wih = path / f"W_ih_{spectral_radius}"
    file_whh = path / f"W_hh_with_{spectral_radius}"
    w_ih = torch.load(file_wih, map_location="cpu")
    w_hh = torch.load(file_whh, map_location="cpu")
    return w_ih, w_hh


#####################################


def generate_new_isolated_weights(expanded_length, hidden_size, Hippo_n_feature,sparsity, spectral_radius, trial, fixed_whh=False, fixed_wih=False, vary_sparsity=False, folder='weights'):
    """
    generate and save new weights for a specific trial
    
    :param expanded_length: size of 'small' network for each feature
    :param hidden_size: network size
    :param Hippo_n_feature: number of features
    :param sparsity: sparsity of the recurrent weights
    :param spectral_radius: spectral radius of the recurrent weights
    :param trial: trial number for which there weights are generated
    :param fixed_whh: True if the recurrent weights are the same for all features
    :param fixed_wih: True if the input weights are the same for all features
    :param vary_sparsity: is the sparsity varied for different weight trials
    :param folder: folder in which the weights should be saved
    """
    if fixed_whh and not fixed_wih:
        w_ih, w_hh = generate_weights_isolated_features_fixed_whh(expanded_length, hidden_size, Hippo_n_feature,sparsity, spectral_radius)
        if vary_sparsity:
            path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights_iso_feat_fixed_whh{trial}_sparsity{sparsity}")
        else:
            path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights_iso_feat_fixed_whh{trial}")
    elif fixed_whh and fixed_wih:
        w_ih, w_hh = generate_weights_isolated_features_fixed_whh_wih(expanded_length, hidden_size, Hippo_n_feature,sparsity, spectral_radius)
        path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/{folder}/custom_weights_iso_feat_fixed_whh_wih{trial}")
    else:
        w_ih, w_hh = generate_weights_isolated_features(expanded_length, hidden_size, Hippo_n_feature,sparsity, spectral_radius)
        path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights_iso_feat{trial}")

    path.mkdir(parents=True, exist_ok=True)
    file_wih = path / f"W_ih_{spectral_radius}"
    file_whh = path / f"W_hh_with_{spectral_radius}"
    torch.save(w_ih, file_wih)
    torch.save(w_hh, file_whh)


def main():
    """
    run the main method, define the parameters at the top and create a for loop over how many weight you want to have (number of weight trials)
    """
    Hippo_n_feature = 16
    Hippo_L = 16
    Hippo_R = 8
    sparsity = 0.2 #[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    spectral_radius = 1.4 #[0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    trial=1
    fixed_whh=True
    fixed_wih=True
    vary_sparsity=False
    folder='weights_shrunk_esn_sr1.4'
    
    expanded_length = Hippo_R + Hippo_L - 1
    hidden_size = Hippo_n_feature * expanded_length

    for t in range(50):
    #for sr in spectral_radius:
        #generate_new_weights(hidden_size, Hippo_n_feature, sparsity, sr, trial=2)
        #for sparse in sparsity:
        generate_new_isolated_weights(expanded_length, hidden_size, Hippo_n_feature,sparsity, spectral_radius, trial=t, fixed_whh=fixed_whh, fixed_wih=fixed_wih, vary_sparsity=vary_sparsity, folder=folder)



if __name__ == "__main__":
    main()