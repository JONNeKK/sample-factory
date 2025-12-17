import torch

from sf_workingdir_lilly.dmlab.custom_rnn import CustomRNN


def test_forward(Hippo_R, Hippo_L, Hippo_n_feature, time_steps=10):
    expanded_length = Hippo_R + Hippo_L - 1
    hidden_size = Hippo_n_feature * expanded_length

    rnn = CustomRNN(input_size=Hippo_n_feature, 
                          hidden_size=hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False,
                          bias=False)
    
    W_ih = torch.zeros( hidden_size, Hippo_n_feature)
    for i in range(Hippo_n_feature):
        for j in range(Hippo_R):
            row = i *  expanded_length + j
            W_ih[row, i] = 1.0

    # weight_hh: shape (hidden_size, hidden_size)
    W_hh = torch.zeros( hidden_size,  hidden_size)
    for i in range(Hippo_n_feature):
        for j in range(1,  expanded_length):
            row = i *  expanded_length + j
            col = i *  expanded_length + (j - 1)
            W_hh[row, col] = 1.0

    with torch.no_grad():
         rnn.weight_ih_l0.copy_(W_ih)
         rnn.weight_hh_l0.copy_(W_hh)
        # rnn.bias_ih_l0.zero_()  was changed in the initialization of the bias object instead
        # rnn.bias_hh_l0.zero_()
        
    # Freeze RNN parameters.
    for param in  rnn.parameters():
        param.requires_grad = False

        # low rank adaptation is learned
    rnn.lr_column.requires_grad = True
    rnn.lr_row.requires_grad = True

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

    return output

def test_lr_updates(Hippo_R, Hippo_L, Hippo_n_feature, time_steps=10):
    expanded_length = Hippo_R + Hippo_L - 1
    hidden_size = Hippo_n_feature * expanded_length

    rnn = CustomRNN(input_size=Hippo_n_feature, 
                          hidden_size=hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False,
                          bias=False)
    
    W_ih = torch.zeros( hidden_size, Hippo_n_feature)
    for i in range(Hippo_n_feature):
        for j in range(Hippo_R):
            row = i *  expanded_length + j
            W_ih[row, i] = 1.0

    # weight_hh: shape (hidden_size, hidden_size)
    W_hh = torch.zeros( hidden_size,  hidden_size)
    for i in range(Hippo_n_feature):
        for j in range(1,  expanded_length):
            row = i *  expanded_length + j
            col = i *  expanded_length + (j - 1)
            W_hh[row, col] = 1.0

    with torch.no_grad():
         rnn.weight_ih_l0.copy_(W_ih)
         rnn.weight_hh_l0.copy_(W_hh)
        # rnn.bias_ih_l0.zero_()  was changed in the initialization of the bias object instead
        # rnn.bias_hh_l0.zero_()
        
    # Freeze RNN parameters.
    for param in  rnn.parameters():
        param.requires_grad = False

        # low rank adaptation is learned
    rnn.lr_column.requires_grad = True
    rnn.lr_row.requires_grad = True

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

if __name__ == "__main__":
    Hippo_R = 2
    Hippo_L = 2
    Hippo_n_feature = 4
    time_steps = 10

    output = test_forward(Hippo_R, Hippo_L, Hippo_n_feature, time_steps)
    print("Output shape:", output.shape)
    print("Output:", output)