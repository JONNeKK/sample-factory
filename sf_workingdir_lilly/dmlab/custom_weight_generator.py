import torch
from torch import Tensor, nn
from pathlib import Path


def generate_weights_pytorch_esn(hidden_size, n_feature, sparsity, spectral_radius):
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
    w_ih = torch.zeros(hidden_size, n_feature)
    w_hh = torch.zeros(hidden_size, hidden_size)

    # fill the weight matrices with weights, only in the positions that correpsond to the correct feature
    for i in range(n_feature):
         wih_current, whh_current = generate_weights_pytorch_esn(expanded_length, 1, sparsity, spectral_radius)
         w_ih[i*expanded_length:i*expanded_length+expanded_length, i] = wih_current.squeeze(-1)
         w_hh[i*expanded_length: i*expanded_length+expanded_length, i*expanded_length: i*expanded_length+expanded_length] = whh_current

    return w_ih, w_hh

def generate_weights_isolated_features_fixed_whh(expanded_length, hidden_size, n_feature, sparsity, spectral_radius):
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
    w_ih = torch.zeros(hidden_size, n_feature)
    w_hh = torch.zeros(hidden_size, hidden_size)

    # fill the weight matrices with weights, only in the positions that correpsond to the correct feature
    wih_current, whh_current = generate_weights_pytorch_esn(expanded_length, 1, sparsity, spectral_radius)
    for i in range(n_feature):
         w_ih[i*expanded_length:i*expanded_length+expanded_length, i] = wih_current.squeeze(-1)
         w_hh[i*expanded_length: i*expanded_length+expanded_length, i*expanded_length: i*expanded_length+expanded_length] = whh_current

    return w_ih, w_hh


####################################

def return_weights_for_spec_rad(spectral_radius, trial=1):
    path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights{trial}")
    path.mkdir(parents=True, exist_ok=True)
    file_wih = path / f"W_ih_{spectral_radius}"
    file_whh = path / f"W_hh_with_{spectral_radius}"
    w_ih = torch.load(file_wih, map_location="cpu")
    w_hh = torch.load(file_whh, map_location="cpu")
    return w_ih, w_hh

def return_weights_for_iso_feat(spectral_radius, trial=1, fixed_whh=False, fixed_wih=False, vary_sparsity=False, sparsity=0.2, folder='weights'):
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

def generate_new_weights(hidden_size, Hippo_n_feature,sparsity, spectral_radius, trial):
    path = Path(f"/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights{trial}")
    path.mkdir(parents=True, exist_ok=True)
    w_ih, w_hh = generate_weights_pytorch_esn(hidden_size, Hippo_n_feature,sparsity, spectral_radius)
    file_wih = path / f"W_ih_{spectral_radius}"
    file_whh = path / f"W_hh_with_{spectral_radius}"
    torch.save(w_ih, file_wih)
    torch.save(w_hh, file_whh)

def generate_new_isolated_weights(expanded_length, hidden_size, Hippo_n_feature,sparsity, spectral_radius, trial, fixed_whh=False, fixed_wih=False, vary_sparsity=False, folder='weights'):
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
    Hippo_n_feature = 16
    Hippo_L = 16
    Hippo_R = 8
    sparsity = 0.2 #[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    spectral_radius = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    trial=1
    fixed_whh=True
    fixed_wih=True
    vary_sparsity=False
    folder='weights_shrunk_esn'
    
    expanded_length = Hippo_R + Hippo_L - 1
    hidden_size = Hippo_n_feature * expanded_length

    #for t in range(50):
    for sr in spectral_radius:
        #generate_new_weights(hidden_size, Hippo_n_feature, sparsity, sr, trial=2)
        #for sparse in sparsity:
        generate_new_isolated_weights(expanded_length, hidden_size, Hippo_n_feature,sparsity, sr, trial=trial, fixed_whh=fixed_whh, fixed_wih=fixed_wih, vary_sparsity=vary_sparsity, folder=folder)

    #w_ih, w_hh = return_weights_for_iso_feat(0.3, trial=1, fixed_whh=True, fixed_wih=False)
    #print(w_ih, w_hh)
    #print(w_ih.size(), w_hh.size())

    #wih, whh = generate_weights_isolated_features_fixed_whh(expanded_length, hidden_size, Hippo_n_feature, sparsity, 0.99)
    #print(wih, whh)
    #print(wih.size(), whh.size())



if __name__ == "__main__":
    main()