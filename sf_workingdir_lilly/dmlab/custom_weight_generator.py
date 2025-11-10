import torch
from torch import Tensor, nn
from pathlib import Path



def generate_weights_pytorch_esn(hidden_size, n_feature, sparsity, spectral_radius):
        w_ih = torch.Tensor(hidden_size, n_feature)
        w_ih.uniform_(-1, 1)
        w_hh = torch.Tensor(hidden_size * hidden_size)
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


def return_weights_for_spec_rad(spectral_radius):
    path = Path("/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights")
    path.mkdir(parents=True, exist_ok=True)
    file_wih = path / f"W_ih_{spectral_radius}"
    file_whh = path / f"W_hh_with_{spectral_radius}"
    w_ih = torch.load(file_wih, map_location="cpu")
    w_hh = torch.load(file_whh, map_location="cpu")
    return w_ih, w_hh


def save_generated_weights(w_ih, w_hh, spectral_radius):
    path = Path("/home/fr/fr_lr554/samplefactory/sample-factory/sf_workingdir_lilly/dmlab/custom_weights")
    path.mkdir(parents=True, exist_ok=True)
    file_wih = path / f"W_ih_{spectral_radius}"
    file_whh = path / f"W_hh_with_{spectral_radius}"
    torch.save(w_ih, file_wih)
    torch.save(w_hh, file_whh)



def main():
    Hippo_n_feature = 16
    Hippo_L = 64 
    Hippo_R = 8
    sparsity = 0.2
    spectral_radius = 0.99
    
    expanded_length = Hippo_R + Hippo_L - 1
    hidden_size = Hippo_n_feature * expanded_length

    wih, whh = generate_weights_pytorch_esn(hidden_size, Hippo_n_feature,sparsity, spectral_radius)
    print(wih, whh)
    print(wih.size(), whh.size())

    save_generated_weights(wih, whh, spectral_radius)

    w_ih, w_hh = return_weights_for_spec_rad(spectral_radius)
    print(w_ih, w_hh)
    print(w_ih.size(), w_hh.size())


if __name__ == "__main__":
    main()