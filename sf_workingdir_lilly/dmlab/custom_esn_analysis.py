import torch
from torch import nn 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


from sf_workingdir_lilly.dmlab.custom_weight_generator import return_weights_for_iso_feat, return_weights_for_spec_rad


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

def get_hidden_states_iso_feat(time_steps, spectral_radius, Hippo_n_feature, Hippo_R, Hippo_L, trial=1, fixed_whh=False, fixed_wih=False, input_idx=0, sparsity=0.2):

    # create the rnn
    W_ih, W_hh = return_weights_for_iso_feat(spectral_radius, trial, fixed_whh, fixed_wih)

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
    # u[:,0,0] = 1.0
    u[:,:,input_idx] = 1.0
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


def get_hidden_states_continuous_input(time_steps, spectral_radius, Hippo_n_feature, Hippo_R, Hippo_L, trial=1):

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
    u[:,:,0] = 1.0
    h0 = torch.zeros(1,1,hidden_size)
    y, h = rnn.forward(u, h0)
    output[0,:] = h

    for i in range(1, time_steps):
        u = torch.zeros(1,1,Hippo_n_feature)
        input_idx = i % Hippo_n_feature
        u[:,:,input_idx] = 1.0
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

def compute_angle_between_hidden_states(hs, time_steps):  # hs: (time steps, hidden size)
    angles = np.zeros(time_steps-1)
    for i in range(time_steps-1):
        v1 = hs[i]
        v2 = hs[i+1]
        dot = v1.dot(v2)
        norm_v1 = v1.norm()
        norm_v2 = v2.norm()

        #print('dot ', dot)
        d = (norm_v1 * norm_v2).clamp(min=1e-20)

        r = dot / d
        angles[i]=torch.acos(r)
        #print(d)
        #print(torch.acos(r))

    return angles

def compute_angle_plane_hidden_state(hs, time_steps):  # hs: (time steps, hidden size)
    angles = np.zeros(time_steps-2)
    for i in range(1, time_steps-1):
        v1 = hs[i-1]
        v2 = hs[i]
        v3 = hs[i+1]

        orth_v = get_orthogonal_vector(v1, v2)

        dot = orth_v.dot(v3)
        norm_orth = orth_v.norm()
        norm_v3 = v3.norm()

        #print('dot ', dot)
        d = (norm_orth * norm_v3).clamp(min=1e-20)

        r = dot / d
        angles[i-1]=torch.acos(r)
        #print(d)
        #print(torch.acos(r))

    return angles

def compute_angle_hypperplane_hidden_state(hs, time_steps):
    angles = np.zeros(time_steps-1)
    basis_vectors = []
    dim = hs.shape[1]

    for i in range(time_steps-1):
        v_current = hs[i]
        v_next = hs[i+1]

        basis_vectors = update_basis_vectors(basis_vectors, v_current)
        v_orth = get_orth_to_basis_vecotr(basis_vectors, dim)

        if torch.norm(v_orth) < 1e-10:
            angles[i] = 0
        else:
            dot = v_orth.dot(v_next)
            norm_orth = v_orth.norm()
            norm_next = v_next.norm()

            d = (norm_orth*norm_next).clamp(min=1e-20)
            r = dot/d
            angles[i] = torch.acos(r)
    return angles


def update_basis_vectors(basis_vectors, v_current):
    v_new = v_current.flatten()
    for v in basis_vectors:
        proj = torch.dot(v_new, v)*v
        v_new = v_new - proj
    v_new_norm = torch.norm(v_new)
    if v_new_norm > 1e-10:
        basis_vectors.append(v_new/v_new_norm)
    return basis_vectors

def get_orth_to_basis_vecotr(basis_vectors, dim):
    v_rand = torch.randn(dim)
    for v in basis_vectors:
        proj = torch.dot(v_rand, v)*v
        v_rand = v_rand - proj
    norm_v_rand = torch.norm(v_rand)
    if norm_v_rand < 1e-10:
        return torch.zeros(dim)
    else:
        return v_rand / norm_v_rand
    

def len_new_orth_vec(hs, time_steps) -> np.ndarray:
    """
    return: orthogonal component at every time step, of all previous hidden states (time steps, hidden size)
    hs: (time steps, hidden size)
    """
    hs = torch.transpose(hs, 0, 1)  # (time steps, hidden size) -> (hidden size, time steps)
    subspace=hs[:,[0]]
    ortho_components = []
    for i in range(1, time_steps):
        current_hidden = hs[:,i]
        current_hidden = current_hidden/torch.norm(current_hidden)
        proj = subspace @ torch.pinverse(subspace) @ current_hidden
        ortho_component = current_hidden - proj
        ortho_components.append(ortho_component)
        ortho_component_norm = ortho_component/torch.norm(ortho_component)
        subspace = torch.cat((subspace, ortho_component_norm.unsqueeze(-1)), dim=1) if ortho_component.norm()>1e-10 else subspace
    ortho_components = np.array(ortho_components) 
    return np.linalg.norm(ortho_components, axis=1)


def get_orthogonal_vector(v1, v2):
    '''
    return: vector that is orthogonal to both of the input vectors
    this is used to check whether a third vector v3 has a non 90° angle to the orthogonal vector, if the angle is 90° to the orthogonal vector
    -> the vector is in the plane spanned by the previous two vectors -> not much new dimensionality is explored compared to the previous vectors
    '''
    v1 = v1.flatten()
    v2 = v2.flatten()
    dim = v1.shape[0]

    v1_norm = torch.norm(v1)
    if v1_norm < 1e-10:
        #print('Value error')
        return torch.zeros(dim)  # TODO checken ob das funktioniert, dass wenn keine aktivität mehr is ein zero vektor zurückgegeben wird und dann der winkel natürlich auch nicht mehr passt
    u1 = v1 / v1_norm

    project_v2_on_u1 = torch.dot(v2, u1) * u1
    u2 = v2 - project_v2_on_u1

    u2_norm = torch.norm(u2)
    if u2_norm < 1e-10:
        u2 = torch.zeros_like(v1)
    else:
        u2 = u2/u2_norm

    v_rand = torch.randn(dim)

    project_on_u1 = torch.dot(v_rand, u1)*u1
    project_on_u2 = torch.dot(v_rand, u2)*u2 if torch.norm(u2)>0 else 0

    v_orth = v_rand - project_on_u1 - project_on_u2
    v_orth_norm = torch.norm(v_orth)
    return v_orth/v_orth_norm


def get_time_of_threshold_crossing(len_ortho_vec, threshold=0.1):
    """
    len_ortho_vec: (trials, Hippo_n_feature, time_steps-1)
    return: time step for each trial at which the length of the orthogonal vector crossed the threshold
    """
    thresh_crossed = np.zeros(len_ortho_vec.shape[0])
    for t in range(len_ortho_vec.shape[0]):
        thresh_crossed[t] = np.where(len_ortho_vec[t,0,:] < threshold)[0][0]
    return thresh_crossed
        



def main():
    spectral_radius = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99]

    hs = get_hidden_states_iso_feat(70, 0.5, 16, 8, 64, trial=1, fixed_whh=True)  

    #hs = torch.Tensor([[1,2,3,4,5],[6,7,8,9,0]])
    len_orth_vec = np.array([[[1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]], [[1, 0.8, 0.6, 0.4, 0.1, 0.09, 0.05]], [[1, 0.8, 0.6, 0.1, 0.02, 0.01, 0.005]]])
    #print(len_orth_vec.shape)
    thresh_crossed = get_time_of_threshold_crossing(len_orth_vec)
    print(thresh_crossed)
    

if __name__ == "__main__":
    main()

