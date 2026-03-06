import torch
from torch import Tensor, nn
import echotorch.nn as echonn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

from types import SimpleNamespace

from logging import warning

from sample_factory.model.core import ModelCore


class FixedESNWithBypassCoreTest(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. 
                              The first Hippo_n_feature dimensions are fed into the fixed RNN,
                              and the remaining (if any) are passed through as bypass features.
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        #self.bypass_size = input_size - self.Hippo_n_feature
        #log.debug("bypass size: {self.bypass_size}")
        # The total register length.
        self.expanded_length = self.R + self.L - 1  
        # The flattened hidden state dimension (RNN core output).
        self.core_output_size = self.Hippo_n_feature * self.expanded_length #+ self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        self.sparsity = 0.2

        # Create a one-layer RNN with ReLU activation.
        
        self.rnn = nn.RNN(input_size=self.n_feature, 
                          hidden_size=self.hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False,
                          bias=False)
        
        # Create the ESN
        self.esn = echonn.ESN(input_dim=self.n_feature,
                         hidden_dim=self.hidden_size,
                         output_dim=self.core_output_size,
                          w_sparsity=self.sparsity,
                          nonlin_func=torch.relu)
        
        # do this so that the weights are not nan or inf
        while self.esn.w.isnan().any():
            self.esn = echonn.ESN(input_dim=self.n_feature,
                                    hidden_dim=self.hidden_size,
                                    output_dim=self.core_output_size,
                                    w_sparsity=self.sparsity,
                                    nonlin_func=torch.relu)

        # Create fixed weight matrices with echotorch ESN class.
        # weight_ih: shape (hidden_size, n_feature)
        '''
        W_ih = torch.zeros(self.hidden_size, self.n_feature)
        for i in range(self.n_feature):
            for j in range(self.R):
                row = i * self.expanded_length + j
                W_ih[row, i] = 1.0
        '''
        W_ih = self.esn.w_in

        # weight_hh: shape (hidden_size, hidden_size)
        '''
        W_hh = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(self.n_feature):
            for j in range(1, self.expanded_length):
                row = i * self.expanded_length + j
                col = i * self.expanded_length + (j - 1)
                W_hh[row, col] = 1.0
        '''
        W_hh = self.esn.w

        W_ih = self.esn.w_in.detach()
        W_hh = self.esn.w.detach()

        # ensure device/dtype match
        W_ih = W_ih.to(self.rnn.weight_ih_l0.device, dtype=self.rnn.weight_ih_l0.dtype)
        W_hh = W_hh.to(self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)

        # Assign the fixed weights and zero out biases.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            #self.rnn.bias_ih_l0.zero_()
            #self.rnn.bias_hh_l0.zero_()
        
        # Freeze RNN parameters.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (concat_output, new_rnn_states) where:
              - concat_output is the concatenation of the fixed RNN output and the bypass features.
              - new_rnn_states is the updated recurrent state.
        """
        # Prepare initial hidden state for RNN.
        # log.info(rnn_states.size())
        h0 = rnn_states.unsqueeze(0)[:,:, :self.hidden_size].contiguous()
        
        
        if isinstance(head_output, PackedSequence):
            # For PackedSequence, work on the underlying data.
            # Split into RNN and bypass parts.
            rnn_data = head_output.data[:, :self.n_feature]
            bypass_data = head_output.data[:, self.n_feature:] if self.bypass_size > 0 else None

            # Create a PackedSequence for the RNN input.
            rnn_packed = PackedSequence(rnn_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
            # Run the RNN.
            rnn_output_packed, new_hidden = self.rnn(rnn_packed, h0)
            new_hidden = new_hidden.squeeze(0)  # shape: (B, core_output_size)
            
            # If bypass features exist, concatenate them.
            if bypass_data is not None:
                # Concatenate along the feature dimension.
                concatenated_data = torch.cat([rnn_output_packed.data, bypass_data], dim=1)

                bypass_data_packed = PackedSequence(bypass_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
                # Assume 'packed' is your PackedSequence and you used batch_first=True when packing.
                padded, lengths = pad_packed_sequence(bypass_data_packed, batch_first=True)

                # For each sequence in the batch, pick the last valid time step.
                # lengths is a tensor of the original sequence lengths.
                last_inputs = padded[torch.arange(padded.size(0)), lengths - 1, :]
                concatenated_data_hidden = torch.cat([new_hidden.data, last_inputs], dim=1)

                # # Compute indices in the packed data that correspond to the last time step of each sequence.
                # last_indices = head_output.batch_sizes.cumsum(0) - 1

                # # Use these indices to index into the bypass_data tensor.
                # last_bypass = bypass_data[last_indices,:]

                # # If the sequences were originally unsorted, restore the original order:
                # last_bypass = last_bypass[head_output.unsorted_indices,:]
                # concatenated_data_hidden = torch.cat([new_hidden.data, last_bypass], dim=1)

            else:
                concatenated_data = rnn_output_packed.data
                concatenated_data_hidden = new_hidden.data

            # Build a new PackedSequence with the concatenated data.
            concat_output = PackedSequence(concatenated_data,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            
            concat_hidden = PackedSequence(concatenated_data_hidden,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            return concat_output, concat_hidden
        else:
            # For Tensor input.
            # Split the input into RNN and bypass parts.
            rnn_input = head_output[:, :self.n_feature]  # shape: (B, n_feature)
            bypass_output = head_output[:, self.n_feature:] if self.bypass_size > 0 else None
            
            # Add sequence dimension for the RNN.
            rnn_input = rnn_input.unsqueeze(0)  # shape: (1, B, n_feature)
            rnn_output, new_hidden = self.rnn(rnn_input, h0)
            new_hidden = new_hidden.squeeze(0)   # shape: (B, core_output_size)
            rnn_output = rnn_output.squeeze(0)     # shape: (B, core_output_size)
            
            # Concatenate the RNN output with bypass features.
            if bypass_output is not None:
                concat_output = torch.cat([rnn_output, bypass_output], dim=1)
                concat_hidden = torch.cat([new_hidden, bypass_output], dim=1)
            else:
                concat_output = rnn_output

                concat_hidden = new_hidden
            
            return concat_output, concat_hidden


# use the logic from the pytorch-esn package to generate weights
class FixedESNWithBypassCoreTestWithCustomWeightsESN(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. 
                              The first Hippo_n_feature dimensions are fed into the fixed RNN,
                              and the remaining (if any) are passed through as bypass features.
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        #self.bypass_size = input_size - self.Hippo_n_feature
        #log.debug("bypass size: {self.bypass_size}")
        # The total register length.
        self.expanded_length = self.R + self.L - 1  
        # The flattened hidden state dimension (RNN core output).
        self.core_output_size = self.Hippo_n_feature * self.expanded_length #+ self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        self.sparsity = 0.8
        self.spectral_radius = 0.98

        # Create a one-layer RNN with ReLU activation.
        
        self.rnn = nn.RNN(input_size=self.n_feature, 
                          hidden_size=self.hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False,
                          bias=False)
        
        W_ih, W_hh = self.generate_weights_pytorch_esn()

        while W_hh.isnan().any() or W_hh.isinf().any() or W_ih.isnan().any() or W_ih.isinf().any():
            W_ih, W_hh = self.generate_weights_pytorch_esn()

        W_ih = W_ih.detach()
        W_hh = W_hh.detach()

        # ensure device/dtype match
        W_ih = W_ih.to(self.rnn.weight_ih_l0.device, dtype=self.rnn.weight_ih_l0.dtype)
        W_hh = W_hh.to(self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)

        # Assign the fixed weights and zero out biases.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            #self.rnn.bias_ih_l0.zero_()
            #self.rnn.bias_hh_l0.zero_()
        
        # Freeze RNN parameters.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (concat_output, new_rnn_states) where:
              - concat_output is the concatenation of the fixed RNN output and the bypass features.
              - new_rnn_states is the updated recurrent state.
        """
        # Prepare initial hidden state for RNN.
        # log.info(rnn_states.size())
        h0 = rnn_states.unsqueeze(0)[:,:, :self.hidden_size].contiguous()
        
        
        if isinstance(head_output, PackedSequence):
            # For PackedSequence, work on the underlying data.
            # Split into RNN and bypass parts.
            rnn_data = head_output.data[:, :self.n_feature]
            bypass_data = head_output.data[:, self.n_feature:] if self.bypass_size > 0 else None

            # Create a PackedSequence for the RNN input.
            rnn_packed = PackedSequence(rnn_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
            # Run the RNN.
            rnn_output_packed, new_hidden = self.rnn(rnn_packed, h0)
            new_hidden = new_hidden.squeeze(0)  # shape: (B, core_output_size)
            
            # If bypass features exist, concatenate them.
            if bypass_data is not None:
                # Concatenate along the feature dimension.
                concatenated_data = torch.cat([rnn_output_packed.data, bypass_data], dim=1)

                bypass_data_packed = PackedSequence(bypass_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
                # Assume 'packed' is your PackedSequence and you used batch_first=True when packing.
                padded, lengths = pad_packed_sequence(bypass_data_packed, batch_first=True)

                # For each sequence in the batch, pick the last valid time step.
                # lengths is a tensor of the original sequence lengths.
                last_inputs = padded[torch.arange(padded.size(0)), lengths - 1, :]
                concatenated_data_hidden = torch.cat([new_hidden.data, last_inputs], dim=1)

                # # Compute indices in the packed data that correspond to the last time step of each sequence.
                # last_indices = head_output.batch_sizes.cumsum(0) - 1

                # # Use these indices to index into the bypass_data tensor.
                # last_bypass = bypass_data[last_indices,:]

                # # If the sequences were originally unsorted, restore the original order:
                # last_bypass = last_bypass[head_output.unsorted_indices,:]
                # concatenated_data_hidden = torch.cat([new_hidden.data, last_bypass], dim=1)

            else:
                concatenated_data = rnn_output_packed.data
                concatenated_data_hidden = new_hidden.data

            # Build a new PackedSequence with the concatenated data.
            concat_output = PackedSequence(concatenated_data,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            
            concat_hidden = PackedSequence(concatenated_data_hidden,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            return concat_output, concat_hidden
        else:
            # For Tensor input.
            # Split the input into RNN and bypass parts.
            rnn_input = head_output[:, :self.n_feature]  # shape: (B, n_feature)
            bypass_output = head_output[:, self.n_feature:] if self.bypass_size > 0 else None
            
            # Add sequence dimension for the RNN.
            rnn_input = rnn_input.unsqueeze(0)  # shape: (1, B, n_feature)
            rnn_output, new_hidden = self.rnn(rnn_input, h0)
            new_hidden = new_hidden.squeeze(0)   # shape: (B, core_output_size)
            rnn_output = rnn_output.squeeze(0)     # shape: (B, core_output_size)
            
            # Concatenate the RNN output with bypass features.
            if bypass_output is not None:
                concat_output = torch.cat([rnn_output, bypass_output], dim=1)
                concat_hidden = torch.cat([new_hidden, bypass_output], dim=1)
            else:
                concat_output = rnn_output

                concat_hidden = new_hidden
            
            return concat_output, concat_hidden
        
    def generate_weights_pytorch_esn(self):
        w_ih = torch.Tensor(self.hidden_size, self.n_feature)
        w_ih.uniform_(-1, 1)
        w_hh = torch.Tensor(self.hidden_size * self.hidden_size)
        w_hh.uniform_(-1, 1)

        # add sparcity to the recurrent matrix
        if self.sparsity<1:
            zero_weights = torch.randperm(int(self.hidden_size * self.hidden_size))
            zero_weights = zero_weights[:int(self.hidden_size * self.hidden_size * (1 - self.sparsity))]
            w_hh[zero_weights] = 0

        # reshape & scale to the desired spectral radius
        w_hh = w_hh.view(self.hidden_size, self.hidden_size)
        print('eigenvals',torch.linalg.eigvals(w_hh))
        abs_eigs = torch.abs(torch.linalg.eigvals(w_hh))
        w_hh = w_hh * (self.spectral_radius / torch.max(abs_eigs))

        return w_ih, w_hh

# use the logic from the echotorch package to generate weights
class FixedESNWithBypassCoreTestWithCustomWeightsEchotorch(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. 
                              The first Hippo_n_feature dimensions are fed into the fixed RNN,
                              and the remaining (if any) are passed through as bypass features.
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        #self.bypass_size = input_size - self.Hippo_n_feature
        #log.debug("bypass size: {self.bypass_size}")
        # The total register length.
        self.expanded_length = self.R + self.L - 1  
        # The flattened hidden state dimension (RNN core output).
        self.core_output_size = self.Hippo_n_feature * self.expanded_length #+ self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        self.sparsity = 0.2
        self.spectral_radius = 0.9

        # Create a one-layer RNN with ReLU activation.
        
        self.rnn = nn.RNN(input_size=self.n_feature, 
                          hidden_size=self.hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False,
                          bias=False)
        
        W_ih, W_hh = self.generate_weights_echotorch()
        

        while W_hh.isnan().any() or W_hh.isinf().any() or W_ih.isnan().any() or W_ih.isinf().any():
            W_ih, W_hh = self.generate_weights_echotorch()

        W_ih = W_ih.detach()
        W_hh = W_hh.detach()

        # ensure device/dtype match
        W_ih = W_ih.to(self.rnn.weight_ih_l0.device, dtype=self.rnn.weight_ih_l0.dtype)
        W_hh = W_hh.to(self.rnn.weight_hh_l0.device, dtype=self.rnn.weight_hh_l0.dtype)

        # Assign the fixed weights and zero out biases.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            #self.rnn.bias_ih_l0.zero_()
            #self.rnn.bias_hh_l0.zero_()
        
        # Freeze RNN parameters.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (concat_output, new_rnn_states) where:
              - concat_output is the concatenation of the fixed RNN output and the bypass features.
              - new_rnn_states is the updated recurrent state.
        """
        # Prepare initial hidden state for RNN.
        # log.info(rnn_states.size())
        h0 = rnn_states.unsqueeze(0)[:,:, :self.hidden_size].contiguous()
        
        
        if isinstance(head_output, PackedSequence):
            # For PackedSequence, work on the underlying data.
            # Split into RNN and bypass parts.
            rnn_data = head_output.data[:, :self.n_feature]
            bypass_data = head_output.data[:, self.n_feature:] if self.bypass_size > 0 else None

            # Create a PackedSequence for the RNN input.
            rnn_packed = PackedSequence(rnn_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
            # Run the RNN.
            rnn_output_packed, new_hidden = self.rnn(rnn_packed, h0)
            new_hidden = new_hidden.squeeze(0)  # shape: (B, core_output_size)
            
            # If bypass features exist, concatenate them.
            if bypass_data is not None:
                # Concatenate along the feature dimension.
                concatenated_data = torch.cat([rnn_output_packed.data, bypass_data], dim=1)

                bypass_data_packed = PackedSequence(bypass_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
                # Assume 'packed' is your PackedSequence and you used batch_first=True when packing.
                padded, lengths = pad_packed_sequence(bypass_data_packed, batch_first=True)

                # For each sequence in the batch, pick the last valid time step.
                # lengths is a tensor of the original sequence lengths.
                last_inputs = padded[torch.arange(padded.size(0)), lengths - 1, :]
                concatenated_data_hidden = torch.cat([new_hidden.data, last_inputs], dim=1)

                # # Compute indices in the packed data that correspond to the last time step of each sequence.
                # last_indices = head_output.batch_sizes.cumsum(0) - 1

                # # Use these indices to index into the bypass_data tensor.
                # last_bypass = bypass_data[last_indices,:]

                # # If the sequences were originally unsorted, restore the original order:
                # last_bypass = last_bypass[head_output.unsorted_indices,:]
                # concatenated_data_hidden = torch.cat([new_hidden.data, last_bypass], dim=1)

            else:
                concatenated_data = rnn_output_packed.data
                concatenated_data_hidden = new_hidden.data

            # Build a new PackedSequence with the concatenated data.
            concat_output = PackedSequence(concatenated_data,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            
            concat_hidden = PackedSequence(concatenated_data_hidden,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            return concat_output, concat_hidden
        else:
            # For Tensor input.
            # Split the input into RNN and bypass parts.
            rnn_input = head_output[:, :self.n_feature]  # shape: (B, n_feature)
            bypass_output = head_output[:, self.n_feature:] if self.bypass_size > 0 else None
            
            # Add sequence dimension for the RNN.
            rnn_input = rnn_input.unsqueeze(0)  # shape: (1, B, n_feature)
            rnn_output, new_hidden = self.rnn(rnn_input, h0)
            new_hidden = new_hidden.squeeze(0)   # shape: (B, core_output_size)
            rnn_output = rnn_output.squeeze(0)     # shape: (B, core_output_size)
            
            # Concatenate the RNN output with bypass features.
            if bypass_output is not None:
                concat_output = torch.cat([rnn_output, bypass_output], dim=1)
                concat_hidden = torch.cat([new_hidden, bypass_output], dim=1)
            else:
                concat_output = rnn_output

                concat_hidden = new_hidden
            
            return concat_output, concat_hidden
        
    def generate_weights_echotorch(self):
        """
        Generate W matrix
        :param output_dim:
        :param w_sparsity:
        :return:
        """
        # Sparsity
        if self.sparsity is None:
            w = torch.rand(self.hidden_size, self.hidden_size) * 2.0 - 1.0
            
        else:
            w = np.random.choice([0.0, 1.0], (self.hidden_size, self.hidden_size),
                                 p=[1.0 - self.sparsity, self.sparsity])
            w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
            w = torch.from_numpy(w.astype(np.float32))

            # w_in = np.random.choice(np.append([0], [1.0, -1.0]),(self.hidden_size, self.n_feature),p=np.append([1.0 - self.sparsity],
             #                                                            [self.sparsity / len([1.0, -1.0])] * len([1.0, -1.0])))
            # w_in = torch.from_numpy(w_in.astype(np.float32))

            # Return
            # return w
        w_in = np.random.randint(0, 2, (self.hidden_size, self.n_feature)) * 2.0 - 1.0
        w_in = torch.from_numpy(w_in.astype(np.float32))
        # end if
        return w_in, w
    # end generate_w


# create a class to test the echotorch esn
# usable now, BUT use the modified echotorch package here: 
# /home/fr/fr_lr554/.conda/envs/env/lib/python3.10/site-packages/echotorch
class TestESN():
    def __init__(self, R, L, Hippo_n_feature, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. 
                              The first Hippo_n_feature dimensions are fed into the fixed RNN,
                              and the remaining (if any) are passed through as bypass features.
        """
        # super().__init__(cfg)
        # Use configuration or defaults.
        self.R = R
        self.L = L
        self.Hippo_n_feature = Hippo_n_feature

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        #self.bypass_size = input_size - self.Hippo_n_feature
        #log.debug("bypass size: {self.bypass_size}")
        # The total register length.
        self.expanded_length = self.R + self.L - 1  
        # The flattened hidden state dimension (RNN core output).
        self.core_output_size = self.Hippo_n_feature * self.expanded_length #+ self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        # Create the ESN
        self.esn = echonn.ESN(input_dim=self.n_feature,
                         hidden_dim=self.hidden_size,
                         output_dim=self.core_output_size,
                          w_sparsity=0.2)
    

# uses the pytorch-esn library  weight logic
class ESNWeights(): 
    def __init__(self, R, L, Hippo_n_feature, input_size):
        self.R = R
        self.L = L
        self.Hippo_n_feature = Hippo_n_feature
        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.expanded_length = self.R + self.L - 1  
        self.core_output_size = self.Hippo_n_feature * self.expanded_length #+ self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        # create the ESN weights by hand like in the pytorch-esn library
        self.sparsity = 0.2
        self.spectral_radius = 0.9

        self.w_ih_esn, self.w_hh_esn = self.generate_weights_pytorch_esn()
        self.w_ih_echo, self.w_hh_echo = self.generate_weights_echotorch()

    

    def generate_weights_echotorch(self):
        """
        Generate W matrix
        :param output_dim:
        :param w_sparsity:
        :return:
        """
        # Sparsity
        if self.sparsity is None:
            w = torch.rand(self.hidden_size, self.hidden_size) * 2.0 - 1.0
            
        else:
            w = np.random.choice([0.0, 1.0], (self.hidden_size, self.hidden_size),
                                 p=[1.0 - self.sparsity, self.sparsity])
            w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
            w = torch.from_numpy(w.astype(np.float32))

            # w_in = np.random.choice(np.append([0], [1.0, -1.0]),(self.hidden_size, self.n_feature),p=np.append([1.0 - self.sparsity],
             #                                                            [self.sparsity / len([1.0, -1.0])] * len([1.0, -1.0])))
            # w_in = torch.from_numpy(w_in.astype(np.float32))

            # Return
            # return w
        w_in = np.random.randint(0, 2, (self.hidden_size, self.n_feature)) * 2.0 - 1.0
        w_in = torch.from_numpy(w_in.astype(np.float32))
        # end if
        return w, w_in
    # end generate_w
        


    def generate_weights_pytorch_esn(self):
        w_ih = torch.Tensor(self.hidden_size, self.n_feature)
        w_ih.uniform_(-1, 1)
        w_hh = torch.Tensor(self.hidden_size * self.hidden_size)
        w_hh.uniform_(-1, 1)

        # add sparcity to the recurrent matrix
        if self.sparsity<1:
            zero_weights = torch.randperm(int(self.hidden_size * self.hidden_size))
            zero_weights = zero_weights[:int(self.hidden_size * self.hidden_size * (1 - self.sparsity))]
            w_hh[zero_weights] = 0

        # reshape & scale to the desired spectral radius
        w_hh = w_hh.view(self.hidden_size, self.hidden_size)
        abs_eigs = torch.abs(torch.linalg.eigvals(w_hh))
        w_hh = w_hh * (self.spectral_radius / torch.max(abs_eigs))

        return w_ih, w_hh


###################################################################################
############################### Tests #############################################
###################################################################################

def make_cfg(R, L, Hippo_n_feature, **extra):
    """
    Helper method for the test methods to create a basic configuration.
    """
    expanded_length = R + L - 1
     # rnn_size is often read by cores; set a sensible default
    defaults = dict(
        Hippo_R=R,
        Hippo_L=L,
        Hippo_n_feature=Hippo_n_feature,
        rnn_size=Hippo_n_feature * expanded_length,
        #nonlinearity='relu',
        device='cpu',
        core_name='BypassFixedESN',
    )
    defaults.update(extra)
    return SimpleNamespace(**defaults)


######################## FixedESNWithBypassCoreTest ###############################


def test_esn_rnn_hidden(R, L, n_feat, iter):
    """
    This method creates an esn with the ecotorch package and an rnn from the pytorch package and lets you do the zero input test to check whether the reservoir gets updated correctly.
    Args: 
        n_feat: number of features (in the DG)
        iter: number of iterations the zero input test should do
    """
    cfg = make_cfg(R=R, L=L, Hippo_n_feature=n_feat)
    # test whether the esn and rnn behave the same when the have the same weights
    test2 = FixedESNWithBypassCoreTest(cfg,n_feat)
    esn = test2.esn
    rnn = test2.rnn

    print(esn.hidden)
    print(esn.w)
    print(esn.w_in)
    print(rnn.weight_hh_l0)
    print(rnn.weight_ih_l0)
    
    # zero input test
    B,T = 1,1
    u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
    u[:,0,0]=1.0
    print('u:', u)

    print('esn hidden:' , esn.hidden)
    h2 = esn.esn_cell(u)
    print('esn hidden:', h2)

    h0 = torch.zeros(1,1,L)
    y, h = rnn.forward(u, h0)
    print('rnn hidden:',h)
    # print('rnn hidden size:',h.size())

    for i in range(iter):
        print('------------iteration', i)
        u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
        print('u:', u)
        
        h2 = esn.esn_cell(u)
        print('esn hidden',h2)

        #h0 = torch.zeros(L)
        y, h = rnn.forward(u, h)
        print('rnn hidden', h)
        i+=1

    
def test_esn_hidden(R, L, n_feat, iter):
    """
    This method creates an esn with the ecotorch package and lets you do the zero input test to check whether the reservoir gets updated correctly.
    Args: 
        n_feat: number of features (in the DG)
        iter: number of iterations the zero input test should do
    """
    cfg = make_cfg(R=R, L=L, Hippo_n_feature=n_feat)
    # test whether the esn and rnn behave the same when the have the same weights
    test2 = FixedESNWithBypassCoreTest(cfg,n_feat)
    esn = test2.esn

    print(esn.hidden)
    print(esn.w)
    print(esn.w_in)
    
    # zero input test
    B,T = 1,1
    u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
    u[:,0,0]=1.0
    print('u:', u)

    print('esn hidden:' , esn.hidden)
    h2 = esn.esn_cell(u)
    print('esn hidden:', h2)


    for i in range(iter):
        print('------------iteration', i)
        u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
        print('u:', u)
        
        h2 = esn.esn_cell(u)
        print('esn hidden',h2)

        i+=1

#####################FixedESNWithBypassCoreTestWithCustomWeightsESN#########################
def test_esn_with_custom_weihgts(R, L, n_feat, iter):
    cfg = make_cfg(R=R, L=L, Hippo_n_feature=n_feat)
    test_obj = FixedESNWithBypassCoreTestWithCustomWeightsESN(cfg,n_feat)
    rnn = test_obj.rnn
    print('w:', rnn.weight_hh_l0)
    print('w_in:', rnn.weight_ih_l0)

    u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
    u[:,0,0]=1.0
    print('u:', u)

    h0 = torch.zeros(1,1,L)
    y, h = rnn.forward(u, h0)
    print('rnn hidden:',h)


    for i in range(iter):
        print('------------iteration', i)
        u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
        print('u:', u)
        
        y, h = rnn.forward(u, h)
        print('rnn hidden', h)

        i+=1

###################FixedESNWithBypassCoreTestWithCustomWeightsEchotorch#####################
def test_esn_with_custom_weihgts_echotorch(R, L, n_feat, iter):
    cfg = make_cfg(R=R, L=L, Hippo_n_feature=n_feat)
    test_obj = FixedESNWithBypassCoreTestWithCustomWeightsEchotorch(cfg,n_feat)
    rnn = test_obj.rnn
    print('w:', rnn.weight_hh_l0)
    print('w_in:', rnn.weight_ih_l0)

    u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
    u[:,0,0]=1.0
    print('u:', u)

    h0 = torch.zeros(1,1,L)
    y, h = rnn.forward(u, h0)
    print('rnn hidden:',h)


    for i in range(iter):
        print('------------iteration', i)
        u = torch.zeros(1,1,n_feat) # shape: (1, B, n_feature)
        print('u:', u)
        
        y, h = rnn.forward(u, h)
        print('rnn hidden', h)

        i+=1

################################## ESNWeights ##############################################

def compare_weight_creation_methods(R, L, n_feat):
    esn_w = ESNWeights(R,L,n_feat,1) # R, L, Hippo_n_feature, input_size
    print(esn_w.w_ih_esn)
    print(esn_w.w_hh_esn)
    print(esn_w.w_ih_echo)
    print(esn_w.w_hh_echo)

    
       
        
def main():
    n_feat = 1
    L = 4
    R = 1  # TODO update everything when wanting to change R, need to check the logic where the input needs to be applied
    
    n_iterations = 5

    #test_esn_rnn_hidden(R, L, n_feat, n_iterations)

    #compare_weight_creation_methods(R, L, n_feat)

    test_esn_with_custom_weihgts(R,L,n_feat, n_iterations)

    #test_esn_with_custom_weihgts_echotorch(R,L,n_feat, n_iterations)

if __name__ == "__main__":
    main()