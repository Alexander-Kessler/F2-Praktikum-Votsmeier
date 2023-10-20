"""
Calculation of the concentration time curves for an exothermic equilibrium 
reaction A<->B for the cases:
   - Isothermal
   - Adiabatic
   - Polytropic
   
Code written by Alexander Keßler on 06.10.23
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import os

# set default tensor
torch.set_default_dtype(torch.double)

# set matplotlib style
plt.style.use(r'thesis.mplstyle')

class generate_data():
    def __init__(self, inlet_conds, bound_conds, species_conds, reactor_conds):
        """Constructor:
        Args:
            inlet_conds (list): inlet mole fraction [-], temperature [K]
            bound_conds (list): pressure [bar], flow velocity [m s-1]
            species_conds (list): velocity constants [s-1], heat capacity [J K−1 mol−1], 
                                  reaction enthalpy [J mol-1], activation energy [J/mol]
                                  pre-exponential factor [s-1]
            reactor_conds (list): thermal transmittance [J m−3 s−1], cross sectional area [m2], 
                                  reactor length [m], temperature of the bath [K]  
        Params:
            x_A0, x_B0, x_N20 (float): mole fractions [-]
            T0 (int): temperature [K]
            p (int): pressure [bar]
            u (int): flow velocity [m s-1]
            k01 (int): pre-exponential factor [s-1]
            E_A1 (int): activation energy [J/mol]
            c_p (int): heat capacity [J K−1 mol−1]
            H_R (int): reaction enthalpy [J mol-1]
            K600 (int): equilibrium constant at 600 K [-]
            U (int): thermal transmittance [J m−3 s−1]
            A (float): cross sectional area [m2]
            L (int): reactor length [m]
            V_R (float): reactor volume [m3]
            R (float): gas constant [J K-1 mol-1]
            V_dot (float): volumetric flow rate [m^3 s-1]
            T_B (int): temperature of the bath [K]
        """
        
        self.x_A0, self.x_B0, self.x_N20 = inlet_conds[:3]
        self.T0 = inlet_conds[3]
        self.p = bound_conds[0]
        self.u = bound_conds[1]
        self.k01 = species_conds[0]
        self.E_A1 = species_conds[1]
        self.c_p = species_conds[2]
        self.H_R = species_conds[3]
        self.K600 = species_conds[4]
        self.U = reactor_conds[0]
        self.A = reactor_conds[1]
        self.L = reactor_conds[2]
        self.T_B = reactor_conds[3]
        self.V_R = self.A * self.L
        self.R = 8.314472
        self.V_dot = self.u * self.A
    
    def calc_inlet_mole_fractions(self):
        """
        Calculate the inlet ammount of substances from the inlet mole fractions 
        with the ideal gas law.
        
        New Params:
            c_ges (float): total concentration [mol m-3]
            c0_vec (1D-array): species concentrations [mol m-3]
            n_A0, n_B0, n_N20 (float): ammount of substance flow rate [mol s-1]
        """
        
        c_ges = self.p * 1e5/(self.R * self.T0)
        self.c0_vec = c_ges * np.array([self.x_A0, self.x_B0, self.x_N20])
        self.n_A0, self.n_B0, self.n_N20 = self.c0_vec * self.V_dot

    def calc_mole_fractions(self, n_matrix):
        """
        Calculate mole fractions from the ammount of substances.
        
        Args:
            n_matrix (2D-array): mole fraction of species A, B and N2
            
        New Params: 
            x_A, x_B, x_N2 (float): mole fractions [-]
        """
        
        self.x_A = n_matrix[:,0]/np.sum(n_matrix, axis=1)
        self.x_B = n_matrix[:,1]/np.sum(n_matrix, axis=1)
        self.x_N2 = n_matrix[:,2]/np.sum(n_matrix, axis=1)
    
    def calc_reaction_rate(self, T, c_A, c_B):
        """
        Calculation of the total reaction rate .

        Args:
            T (float): temperature [K]
            c_A (float): concentration of species A [mol m-3]
            c_B (float): concentration of species B [mol m-3]
        New Params:
            r_tot (float): total reaction rate [mol m-3 m-1]
        """
        
        # Calculate velocity constant with Arrhenius
        k_hin = self.k01 * math.exp(-self.E_A1 / (self.R * T))
        # Calculate reaction rates
        r_hin = k_hin * c_A
        self.r_tot = r_hin * (1 - ((c_B / c_A) / self.K600))

    def isotherm(y, z, self):
        """
        ODEs from the mass balances for the ODE-Solver to solve an isothermal 
        plug flow reactor.
        
        New Params: 
            y (list): ammount of substance from species A, B and C [mol],
                      temperature [K]
            dn_dz_A, dn_dz_B (float): differentials from the ammount of 
                                      substances [mol m-1 s-1]
        """
        # Calculate the total reaction rate
        generate_data.calc_reaction_rate(self, y[3], y[0]/self.V_dot, y[1]/self.V_dot)

        dn_dz_A = self.A * (-self.r_tot)
        dn_dz_B = self.A * (+self.r_tot)
        dn_dz_N2 = 0
        dT_dz = 0
        
        return [dn_dz_A, dn_dz_B, dn_dz_N2, dT_dz]
        
    def adiabatic(y, z, self):
        """
        ODEs from the mass balances and heat balance for the ODE-Solver to 
        solve an adiabatic plug flow reactor.
        
        New Params: 
            y (list): ammount of substance from species A, B and C [mol],
                      temperature [K]
            dn_dz_A, dn_dz_B (float): differentials from the ammount of 
                                      substances [mol m-1 s-1]
            dT_dz (float): differentials from the temperature [K m-1]
        """
        # Calculate the total reaction rate
        generate_data.calc_reaction_rate(self, y[3], y[0]/self.V_dot, y[1]/self.V_dot)

        dn_dz_A = self.A * (-self.r_tot)
        dn_dz_B = self.A * (+self.r_tot)
        dn_dz_N2 = 0
        dT_dz = 1/(self.c_p * (y[0] + y[1] + y[2])) * \
            (-self.A * self.r_tot * self.H_R)
            
        return [dn_dz_A, dn_dz_B, dn_dz_N2, dT_dz]
    
    def polytrop(y, z, self):
        """
        ODEs from the mass balances and heat balance for the ODE-Solver to 
        solve an polytrop plug flow reactor.
        
        New Params: 
            y (list): ammount of substance from species A, B and C [mol],
                      temperature [K]
            dn_dz_A, dn_dz_B (float): differentials from the ammount of 
                                      substances [mol m-1 s-1]
            dT_dz (float): differentials from the temperature [K m-1]
        """
        # Calculate the total reaction rate
        generate_data.calc_reaction_rate(self, y[3], y[0]/self.V_dot, y[1]/self.V_dot)

        dn_dz_A = self.A * (-self.r_tot)
        dn_dz_B = self.A * (+self.r_tot)
        dn_dz_N2 = 0
        dT_dz = 1/(self.c_p * (y[0] + y[1] + y[2])) * \
            (-self.A * self.r_tot * self.H_R - self.A * self.U * (y[3]-self.T_B))

        return [dn_dz_A, dn_dz_B, dn_dz_N2, dT_dz]
    
    def solve_ode(self, thermo_state, reactor_lengths, plot):
        """
        Solution of the ODE from a plug flow reactor. It is possible to choose 
        between the states isothermal, adiabatic and polytropic.

        Args:
            thermo_state (string): 'isotherm', 'adiabatic' or 'polytrop'
            reactor_lengths (1D-array): x-vec with reactor length for the ODE-Solver [m]
            plot (bool): plotting the results

        Returns:
            y (2D-array): ammount of substances [mol], temperature [K]
            n_A, n_B, n_N2 (1D-array): ammount of substances [mol]
            T (1D-array): temperature [K]

        """
        # Calculate the inlet ammount of substances
        generate_data.calc_inlet_mole_fractions(self)
        
        # Solve ODE for isotherm, adiabatic or polytrop reactor
        if thermo_state == 'isotherm':
            y = odeint(generate_data.isotherm, [self.n_A0, self.n_B0, self.n_N20, self.T0], reactor_lengths, args=(self,))
        elif thermo_state == 'adiabatic':
            y = odeint(generate_data.adiabatic, [self.n_A0, self.n_B0, self.n_N20, self.T0], reactor_lengths, args=(self,))
        elif thermo_state == 'polytrop':
            y = odeint(generate_data.polytrop, [self.n_A0, self.n_B0, self.n_N20, self.T0], reactor_lengths, args=(self,))
        else:
            raise Exception('Wrong thermal state! Choose between isotherm, adiabatic or polytrop.')
        
        # Store results
        self.n_A = y[:,0]
        self.n_B = y[:,1]
        self.n_N2 = y[:,2]
        self.T = y[:,3]
        
        # Calculate mole fractions
        generate_data.calc_mole_fractions(self, y[:,:3])
        
        # Plotting results from the ODE-Solver
        if plot:
            generate_data.plot(self, reactor_lengths, thermo_state)
            
    
    def plot(self, reactor_lengths, thermo_state):
        """
        Plotting results from the ODE-Solver.
        """
        plt.figure()
        plt.plot(reactor_lengths, self.x_A, 'b-', label='xA')
        plt.plot(reactor_lengths, self.x_B, 'r-', label='xB')
        plt.xlabel('reactor length')
        plt.ylabel('mole fractions')
        plt.legend()
        plt.title(thermo_state)
        
        plt.figure()
        plt.plot(reactor_lengths, self.T, 'r-', label='temperature')
        plt.xlabel('reactor length')
        plt.ylabel('temperature')
        plt.title(thermo_state)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size_NN, hidden_size_NN, output_size_NN, num_layers_NN, T0):
        # Inheritance of the methods and attributes of the parent class 
        super(NeuralNetwork, self).__init__()
        
        # Defining the hyperparameters of the neural network
        self.input_size = input_size_NN
        self.output_size = output_size_NN
        self.hidden_size  = hidden_size_NN
        self.num_layers = num_layers_NN
        self.layers = torch.nn.ModuleList()
        self.T0 = T0
        
        # Add the first, hidden and last layer to the neural network
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))
    
    # Define the forward pass
    def forward(self, x):     
        # All layers of the neural network are passed through
        for layer in self.layers[:-1]:
            x = layer(x)
            # The activation function tanh is used to introduce non-linearity 
            # into the neural network to solve non-linear problems.
            x = torch.tanh(x)
        x = self.layers[-1](x)
        # The exponential function is used to ensure that the temperatures 
        # are always positive. The multiplication of T0 provides a scaling of 
        # the high temperatures to a value close to 1 -> T/T0.
        x[:,2] = torch.exp(x[:,2]) * self.T0
        
        return x

class PINN_loss(torch.nn.Module):
    def __init__(self, weight_factors, init_conds, bound_conds, species_conds, \
                 reactor_conds, n_N20, epsilon):
        """
            New Args:
                weight_factors (list): weighting factors [-]
                init_conds (tensor): inlet ammount of substances [mol s-1], 
                                     inlet temperature [K]
            New Params:
                w_AB (float): weighting factor of all loss functions which 
                              depends on species A and B
                w_T (float): weighting factor of all loss functions which 
                             depends on the reactor temperature
                w_GE_AB (float): weighting factor from the loss function of 
                                 the governing equation which depends on species 
                                 A and B
                w_GE_T (float): weighting factor from the loss function of 
                                the governing equation which depends on the 
                                reactor temperature
                w_IC_AB (float): weighting factor from the loss function of 
                                 the initial condition which depends on species 
                                 A and B
                w_IC_T (float): weighting factor from the loss function of 
                                the initial condition which depends on the 
                                reactor temperature
                n_N20 (float): initial ammount of substance of N2.
                epsilon (int): causality parameter.
                                
            For the other arguments and parameters, please look in the
            class generate_data().
        """
        super(PINN_loss, self).__init__()
        
        # New parameter
        self.w_AB, self.w_T, self.w_GE_AB, self.w_GE_T, self.w_IC_AB, \
            self.w_IC_T = weight_factors
        self.init_conds = init_conds
        self.n_N20 = n_N20
        self.epsilon = torch.tensor(epsilon)
        
        # Parameter known from the class generate_data()
        self.u = torch.tensor(bound_conds[1])
        self.k01 = torch.tensor(species_conds[0])
        self.E_A1 = torch.tensor(species_conds[1])
        self.c_p = torch.tensor(species_conds[2])
        self.H_R = torch.tensor(species_conds[3])
        self.K600 = torch.tensor(species_conds[4])
        self.U = torch.tensor(reactor_conds[0])
        self.A = torch.tensor(reactor_conds[1])
        self.T_B = torch.tensor(reactor_conds[3])
        self.R = torch.tensor(8.314472)
        self.V_dot = self.u * self.A
    
    def isotherm(self, c_A, c_B, T):
        """
        Calculate ODEs from the mass balances for an isothermal plug flow reactor.
        
        New Params: 
            c_A (tensor): Concentrations of species A. 
            c_B (tensor): concentration of species B [mol m-3]
            T (tensor): temperature [K]
        Returns:
            dn_dz_A, dn_dz_B (tensor): differentials from the ammount of 
                                      substances [mol m-1 s-1]
        """
        
        # Calculate the total reaction rate
        k_hin = self.k01 * torch.exp(-self.E_A1 / (self.R * T)) # 
        r_hin = k_hin * c_A
        r_tot = r_hin * (1 - ((c_B / c_A) / self.K600))
        
        dn_dz_A_pred = self.A * (-r_tot)
        dn_dz_B_pred = self.A * (+r_tot)
        dT_dz_pred = 0
        
        return [dn_dz_A_pred,dn_dz_B_pred,dT_dz_pred]
    
    def adiabatic(self, c_A, c_B, T):
        """
        Calculate ODEs from the mass balances and heat balance for an adiabatic 
        plug flow reactor.
        
        New Params: 
            c_A (tensor): Concentrations of species A. 
            c_B (tensor): concentration of species B [mol m-3]
            T (tensor): temperature [K]
        Returns:
            dn_dz_A, dn_dz_B (tensor): differentials from the ammount of 
                                      substances [mol m-1 s-1]
            dT_dz (tensor): differentials from the temperature [K m-1]
        """
        
        # Calculate the total reaction rate
        k_hin = self.k01 * torch.exp(-self.E_A1 / (self.R * T)) # 
        r_hin = k_hin * c_A
        r_tot = r_hin * (1 - ((c_B / c_A) / self.K600))
        
        dn_dz_A_pred = self.A * (-r_tot)
        dn_dz_B_pred = self.A * (+r_tot)
        dT_dz_pred = 1/(self.c_p * (c_A * self.V_dot + c_B * self.V_dot + \
                    self.n_N20)) * (-self.A * r_tot * self.H_R)
        
        return [dn_dz_A_pred,dn_dz_B_pred,dT_dz_pred]
    
    def polytrop(self, c_A, c_B, T):
        """
        Calculate ODEs from the mass balances and heat balance for an polytrop 
        plug flow reactor.
        
        New Params: 
            c_A (tensor): Concentrations of species A. 
            c_B (tensor): concentration of species B [mol m-3]
            T (tensor): temperature [K]
        Returns:
            dn_dz_A, dn_dz_B (tensor): differentials from the ammount of 
                                      substances [mol m-1 s-1]
            dT_dz (tensor): differentials from the temperature [K m-1]
        """
        
        # Calculate the total reaction rate
        k_hin = self.k01 * torch.exp(-self.E_A1 / (self.R * T)) # 
        r_hin = k_hin * c_A
        r_tot = r_hin * (1 - ((c_B / c_A) / self.K600))
        
        dn_dz_A_pred = self.A * (-r_tot)
        dn_dz_B_pred = self.A * (+r_tot)
        dT_dz_pred = 1/(self.c_p * (c_A * self.V_dot + c_B * self.V_dot + \
                    self.n_N20)) * (-self.A * r_tot * self.H_R - self.A * \
                                    self.U * (T-self.T_B))

        return [dn_dz_A_pred,dn_dz_B_pred,dT_dz_pred]
    
    def calc_IC_loss(self, y_pred, x):
        """
        Calculate the loss function of the initial condition.
        
        Args:
            y_pred (tensor): Predicted output. Here ammount of substances from 
                             species A and B.
            x (tensor): Input values. Here reactor length.
        """
        
        # Calculation of the mean square displacement between the predicted and 
        # original initial conditions
        loss_IC_A = (y_pred[0, 0] - self.init_conds[0])**2
        loss_IC_B = (y_pred[0, 1] - self.init_conds[1])**2
        loss_IC_T = (y_pred[0, 2] - self.init_conds[2])**2
        
        return [loss_IC_A, loss_IC_B, loss_IC_T]
    
    def calc_GE_loss(self, y_pred, x, thermo_state):
        """
        Calculate the loss function of the governing equation.
        
        Args:
            y_pred (tensor): Predicted output. Here ammount of substances from 
                             species A and B.
            x (tensor): Input values. Here reactor length.
            thermo_state (string): 'isotherm', 'adiabatic' or 'polytrop'.
        """
           
        # Calculate the gradients of tensor values
        dn_dz_A = torch.autograd.grad(outputs=y_pred[:, 0], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 0]),
                                retain_graph=True, create_graph=True)[0]
        dn_dz_B = torch.autograd.grad(outputs=y_pred[:, 1], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 1]),
                                retain_graph=True, create_graph=True)[0]        
        dT_dz = torch.autograd.grad(outputs=y_pred[:, 2], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 2]),
                                retain_graph=True, create_graph=True)[0]
        
        # Calculation of the differentials
        c_A = y_pred[:, 0].reshape(-1, 1) / self.V_dot
        c_B = y_pred[:, 1].reshape(-1, 1) / self.V_dot
        T = y_pred[:, 2].reshape(-1, 1)
        
        if thermo_state == 'isotherm':
            dn_dz_A_pred, dn_dz_B_pred, dT_dz_pred = PINN_loss.isotherm(self, c_A, c_B, T)
        elif thermo_state == 'adiabatic':
            dn_dz_A_pred, dn_dz_B_pred, dT_dz_pred = PINN_loss.adiabatic(self, c_A, c_B, T)
        elif thermo_state == 'polytrop':
            dn_dz_A_pred, dn_dz_B_pred, dT_dz_pred = PINN_loss.polytrop(self, c_A, c_B, T)
        else:
            raise Exception('Wrong thermal state! Choose between isotherm, adiabatic or polytrop.')
            
        # Calculation of the mean square displacement between the gradients of 
        # autograd and differentials of the mass and heat balances
        loss_GE_A = (dn_dz_A - dn_dz_A_pred)**2
        loss_GE_B = (dn_dz_B - dn_dz_B_pred)**2
        loss_GE_T = (dT_dz - dT_dz_pred)**2
        
        return [loss_GE_A, loss_GE_B, loss_GE_T]
    
    def causal_training(self, x, loss_IC_A, loss_IC_B, loss_IC_T, loss_GE_A, \
                        loss_GE_B, loss_GE_T):
        """
        Previously, we used torch.mean() to average all the losses and the 
        neural network tried to reduce the gradient of all the points equally. 
        With this function, we implement a new approach in which the points are 
        optimised one after the other by considering weighting factors for the 
        individual points. The idea behind this is that the points are related 
        to each other and the error in the points at the beginning has an 
        influence on the error in the points later and thus accumulates. For 
        this reason, the prediction by the neural network has so far been very 
        good at the beginning and very bad at the end of the points. 
        """
        
        # Form the sum of the losses
        losses = torch.zeros_like(x)
        losses[0,0] = loss_IC_A + loss_IC_B + loss_IC_T + \
            loss_GE_A[0] + loss_GE_B[0] + loss_GE_T[0]
        losses[1:] = loss_GE_A[1:] + loss_GE_B[1:] + loss_GE_T[1:]
        
        # Calculate the weighting factors of the losses
        weight_factors = torch.zeros_like(x)
        for i in range(x.size(0)):
            w_i = torch.exp(-self.epsilon*torch.sum(losses[0:i]))
            weight_factors[i] = w_i
        self.weight_factors = weight_factors
        
        # Consider the weighting factors in the losses
        total_loss = torch.mean(weight_factors * losses)
            
        return total_loss
        
        
    def forward(self, x, y, y_pred, thermo_state):
        '''
        Calculation of the total loss.
        
        Args:
            x (tensor): Input values. Here reactor length.
            y (tensor): Training data. Here ammount of substances from species
                        A and B.
            y_pred (tensor): Predicted output. Here ammount of substances from 
                             species A and B.
            thermo_state (string): 'isotherm', 'adiabatic' or 'polytrop'.
        '''
        
        # Calculation of the total loss
        loss_IC_A, loss_IC_B, loss_IC_T = self.calc_IC_loss(y_pred, x)
        loss_GE_A, loss_GE_B, loss_GE_T = self.calc_GE_loss(y_pred, x, thermo_state)
        
        # Store losses before weighting
        losses_before_weighting = np.array([np.mean(loss_IC_A.detach().numpy()), \
                                    np.mean(loss_IC_B.detach().numpy()), \
                                    np.mean(loss_IC_T.detach().numpy()), \
                                    np.mean(loss_GE_A.detach().numpy()), \
                                    np.mean(loss_GE_B.detach().numpy()), \
                                    np.mean(loss_GE_T.detach().numpy())])
        
        # Consider weighting factors of the loss functions
        loss_IC_A = self.w_IC_AB * self.w_AB * loss_IC_A
        loss_IC_B = self.w_IC_AB * self.w_AB * loss_IC_B
        loss_IC_T = self.w_IC_T * self.w_T * loss_IC_T
        loss_GE_A = self.w_GE_AB * self.w_AB * loss_GE_A
        loss_GE_B = self.w_GE_AB * self.w_AB * loss_GE_B
        loss_GE_T = self.w_IC_T * self.w_T * loss_GE_T
        
        # Calculate the total loss
        total_loss = PINN_loss.causal_training(self, x, loss_IC_A, \
                        loss_IC_B, loss_IC_T, loss_GE_A, loss_GE_B, loss_GE_T)
        
        # Store losses after weighting
        losses_after_weighting = np.array([np.mean(loss_IC_A.detach().numpy()), \
                                   np.mean(loss_IC_B.detach().numpy()), \
                                   np.mean(loss_IC_T.detach().numpy()), \
                                   np.mean(loss_GE_A.detach().numpy()), \
                                   np.mean(loss_GE_B.detach().numpy()), \
                                   np.mean(loss_GE_T.detach().numpy())])
            
        return [total_loss, losses_before_weighting, losses_after_weighting]

def train(x, y, network, calc_loss, optimizer, num_epochs, thermo_state, analytical_solution_x_A, \
          analytical_solution_x_B, n_N20, analytical_solution_T, plot_interval):
    def closure():
        """
        The Closure function is required for the L-BFGS optimiser.
        """
        # Set gradient to zero
        if torch.is_grad_enabled():
            optimizer.zero_grad()
            
        # Forward pass
        ypred = network(x)
        
        # Compute loss
        total_loss, losses_before_weighting, losses_after_weighting = \
            calc_loss(x, y, ypred, thermo_state)
        
        # Backward pass
        if total_loss.requires_grad:
            total_loss.backward()

        return total_loss
    
    def prepare_plots_folder():
        """
        This function is executed at the beginning of the training to prepare 
        the folder with the plots. It checks whether the folder for the plots 
        is existing. If it does not exist, the folder is created. If it is 
        exists, all files in the folder are deleted.
        """
        
        # Path to the folder with the Python script
        script_folder = os.path.dirname(os.path.abspath(__file__))
        folder_name = "plots"
        folder_path = os.path.join(script_folder, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            # Create folder if it does not exist
            os.makedirs(folder_path)
        else:
            # Delete files in the folder if it exists
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Fehler beim Löschen der Datei {file_path}: {e}")
        
        print(f"Der Ordner '{folder_name}' wurde überprüft und gegebenenfalls erstellt bzw. geleert.")
    
    def calculate_msd(y_pred, analytical_solution_x_A, analytical_solution_x_B, \
                      analytical_solution_T):
        analytical_solution_x_A = torch.tensor(analytical_solution_x_A.reshape(-1, 1))
        analytical_solution_x_B = torch.tensor(analytical_solution_x_B.reshape(-1, 1))
        analytical_solution_T = torch.tensor(analytical_solution_T.reshape(-1, 1))
        y_analytical_solution = torch.cat((analytical_solution_x_A, \
                                           analytical_solution_x_B, \
                                               analytical_solution_T), dim=1)
        squared_displacements = (y_analytical_solution-y_pred)**2
        msd = torch.mean(squared_displacements, dim=0).detach().numpy()
            
        return msd
    
    # Prepare folders for the plots 
    prepare_plots_folder()
    
    # Training loop
    loss_values = np.zeros((num_epochs,13))
    msd_NN = np.zeros((num_epochs,3))
    for epoch in range(num_epochs):
        # Updating the network parameters with calculated losses and gradients 
        # from the closure-function
        optimizer.step(closure)

        # Calculation of the current total loss after updating the network parameters 
        # in order to print them in the console afterwards and plot them.
        total_loss, losses_before_weighting, losses_after_weighting = \
            calc_loss(x, y, network(x), thermo_state)
        loss_values[epoch,:] = np.hstack((np.array(total_loss.detach().numpy()),\
                                          losses_before_weighting,losses_after_weighting))

        print('Epoch: ', epoch+1, 'Loss: ', total_loss.item(), 'causal weights sum: ', \
              np.round(np.sum(calc_loss.weight_factors.detach().numpy()),decimals=4))
        
        # Calculate the mean square error between the analytical solution and 
        # the predicted values 
        msd_NN[epoch,:] = calculate_msd(network(x),analytical_solution_x_A, \
                               analytical_solution_x_B, analytical_solution_T)
            
        # Create plots in the given plot_interval
        if (epoch+1)%plot_interval == 0 or epoch == 0:
            y_pred = network(x)
            y_pred = y_pred.detach().numpy()
            y_pred = np.insert(y_pred, 2, n_N20, axis=1)
            y_pred_x_A, y_pred_x_B = calc_mole_frac(y_pred[:,:3])
            y_pred_T = y_pred[:, 3]
                
            plots(x.detach().numpy(),analytical_solution_x_A, \
                  analytical_solution_x_B, analytical_solution_T, \
                     y_pred_x_A, y_pred_x_B, y_pred_T, save_plot=True, \
                        plt_num=epoch+1, weight_factors=calc_loss.weight_factors)        
            
    return [loss_values, msd_NN]
        
def calc_mole_frac(n_matrix):
    """
    Calculate mole fractions from the ammount of substances. Takes into account 
    only two species.
    
    Args:
        n_matrix (2D-array): ammount of substance from species A and B in 
                             dependence of the reactor length
    """
    x_A = n_matrix[:,0]/np.sum(n_matrix, axis=1)
    x_B = n_matrix[:,1]/np.sum(n_matrix, axis=1)
    
    return [x_A,x_B]

def plots(reactor_lengths, analytical_solution_x_A, analytical_solution_x_B, \
          analytical_solution_T,predicted_x_A,predicted_x_B,predicted_T,save_plot=False,\
                plt_num=None,weight_factors=None,loss_values=None,msd_NN=None):
    """
    This function is used to save the plots during the training and to plot 
    them after the training. 

    Args:
        reactor_lengths (1D-array): reactor lengths [m]
        analytical_solution_x_A (1D-array): Analytical solution of the mole fraction of A.
        analytical_solution_x_B (1D-array): Analytical solution of the mole fraction of B.
        analytical_solution_T (1D-array): Analytical solution of the reactor temperature.
        predicted_x_A (1D-array): Network solution of the mole fraction of A.
        predicted_x_B (1D-array): Network solution of the mole fraction of B.
        predicted_T (1D-array): Network solution of the reactor temperature.
        save_plot (boolean): Save the plot in the plots directory.
        plt_num (int or string): This number will be added to the name of the plot.
    """
    # Path to the folder with the plots
    script_folder = os.path.dirname(os.path.abspath(__file__))
    folder_name = "plots"
    folder_path = os.path.join(script_folder, folder_name)
    
    # Mole fractions plot
    plt.figure()
    plt.plot(reactor_lengths, analytical_solution_x_A, 'b-', label=r'$x_{\rm{A}}$')
    plt.scatter(reactor_lengths, predicted_x_A, color = 'g', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{A,pred}}$')
    plt.plot(reactor_lengths, analytical_solution_x_B, 'r-', label=r'$x_{\rm{B}}$')
    plt.scatter(reactor_lengths, predicted_x_B, color = 'y', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{B,pred}}$')
    plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
    plt.ylabel(r'$mole\:fractions$')
    plt.ylim(analytical_solution_x_A[0],analytical_solution_x_A[-1])
    plt.legend(loc='center right')
        
    if save_plot:
        plt.savefig(f'{folder_path}/mole_fraction_{plt_num}.png', dpi=200)
        plt.close()
        
    # Temperature plot
    plt.figure()
    plt.plot(reactor_lengths, analytical_solution_T, 'r-', label=r'$T$')
    plt.scatter(reactor_lengths, predicted_T, color = 'g', s=12, alpha=0.8, \
                edgecolors='b', label=r'$T_{\rm{pred}}$')
    plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
    plt.ylabel(r'$temperature\:/\:\rm{K}$')
    plt.ylim(500,650)
    plt.legend(loc='center right')
    
    if save_plot:
        plt.savefig(f'{folder_path}/temperature_{plt_num}.png', dpi=200)
        plt.close()
        
    # Plot weighting factors from causal training
    if weight_factors is not None:
        plt.figure()
        plt.plot(reactor_lengths, weight_factors.detach().numpy(), 'r-', \
                 label=f'$sum:\:{np.round(np.sum(weight_factors.detach().numpy()),decimals=4)}$')
        plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
        plt.ylabel(r'$causal\:weighting\:factor$')
        plt.ylim(0,1)
        plt.legend(loc='upper right')
        
        if save_plot:
            plt.savefig(f'{folder_path}/causal_weights_{plt_num}.png', dpi=200)
            plt.close()
    
    # Plot losses without weighting factors
    if loss_values is not None:
        plt.figure()
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,0], label=r'$L_{\rm{total}}$')
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,1], label=r'$L_{\rm{IC,A}}$')
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,2], label=r'$L_{\rm{IC,B}}$')
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,3], label=r'$L_{\rm{IC,T}}$')
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,4], label=r'$L_{\rm{GE,A}}$')
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,5], label=r'$L_{\rm{GE,B}}$')
        plt.plot(list(range(1, len(loss_values)+1)), loss_values[:,6], label=r'$L_{\rm{GE,T}}$')
        plt.xlabel(r'$Number\:of\:epochs$')
        plt.ylabel(r'$losses$')
        plt.yscale('log')
        plt.legend(loc='center right')
        
        if save_plot:
            plt.savefig(f'{folder_path}/loss_values_{plt_num}.png', dpi=200)
            plt.close()
    
    # Plot the MSD between the analytical solution and the prediction by the 
    # neural network
    if msd_NN is not None:
        plt.figure()
        plt.plot(list(range(1, len(loss_values)+1)), msd_NN[:,0], label=r'$MSD\rm{(}x_{\rm{A}}\rm{)}$')
        plt.plot(list(range(1, len(loss_values)+1)), msd_NN[:,1], label=r'$MSD\rm{(}x_{\rm{B}}\rm{)}$')
        plt.plot(list(range(1, len(loss_values)+1)), msd_NN[:,2], label=r'$MSD\rm{(}T\rm{)}$')
        plt.xlabel(r'$Number\:of\:epochs$')
        plt.ylabel(r'$MSD$')
        plt.yscale('log')
        plt.legend(loc='center right')
        
        if save_plot:
            plt.savefig(f'{folder_path}/msd_{plt_num}.png', dpi=200)
            plt.close()
        
        
if __name__ == "__main__":
    # Define parameters for the model
    reactor_lengths = np.linspace(0,10,num=100)
    inlet_conds = [0.1,0,0.9,600] #x_A0,x_B0,x_N20,T0
    bound_conds = [1,1] #p,u
    species_conds = [6*1e8,1e5,30,4e4,10] #k1,cp,Hr,E_A,k0
    reactor_conds = [100,0.1,10,600] #U_perv,A_reactor,L_reactor,T_bath
    
    thermo_state = 'polytrop' #'isotherm','adiabatic','polytrop'
    plot_analytical_solution = False #True,False
    
    input_size_NN = 1
    hidden_size_NN = 32
    output_size_NN = 3
    num_layers_NN = 3
    num_epochs = 500
    weight_factors = [1e3,1,1,1,1,1] #w_AB,w_T,w_GE_AB,w_GE_T,w_IC_AB,w_IC_T
    epsilon = 0 #epsilon=0: old model, epsilon!=0: new model, optimized value: 2
    plot_interval = 10 # Plotting during NN-training
    
    # Calculation of the analytical curves
    model = generate_data(inlet_conds, bound_conds, species_conds, reactor_conds)
    model.solve_ode(thermo_state, reactor_lengths, plot=plot_analytical_solution)
    
    analytical_solution_x_A = model.x_A
    analytical_solution_x_B = model.x_B
    analytical_solution_T = model.T

    # Set up the neural network
    network = NeuralNetwork(input_size_NN=input_size_NN, hidden_size_NN=hidden_size_NN,\
                            output_size_NN=output_size_NN, num_layers_NN=num_layers_NN,\
                            T0 = inlet_conds[3])
    optimizer = torch.optim.LBFGS(network.parameters(), lr=1, line_search_fn= \
                                  "strong_wolfe", max_eval=None, tolerance_grad \
                                      =1e-50, tolerance_change=1e-50)
        
    x = torch.tensor(reactor_lengths.reshape(-1, 1), requires_grad=True)
    y = None
    init_conds = torch.tensor([model.n_A0, model.n_B0, model.T0], requires_grad=True)
    
    # Train the neural network
    calc_loss = PINN_loss(weight_factors, init_conds, bound_conds, species_conds, \
                          reactor_conds, model.n_N20*torch.ones_like(x), epsilon) 
    loss_values, msd_NN = \
        train(x, y, network, calc_loss, optimizer, num_epochs, thermo_state, \
              analytical_solution_x_A, analytical_solution_x_B, model.n_N20, \
                 analytical_solution_T, plot_interval)
    
    # Predict data with the trained neural network
    y_pred = network(x)
    # Convert the pytorch tensor to a numpy array
    y_pred = y_pred.detach().numpy()
    # Add the amount of substance from N2 to the the predicted data because N2 
    # is not taken into account in the neural network (N2 stays constant)
    y_pred = np.insert(y_pred, 2, model.n_N20, axis=1)
    # Extract mole fractions
    predicted_x_A, predicted_x_B = calc_mole_frac(y_pred[:,:3])
    predicted_T = y_pred[:, 3]
    
    # Plot the mole fraction, temperature and weighting factors
    plots(reactor_lengths, analytical_solution_x_A, analytical_solution_x_B, \
          analytical_solution_T, predicted_x_A, predicted_x_B, predicted_T, \
          plt_num = f'{num_epochs}_final', save_plot = True, loss_values=loss_values, \
          msd_NN = msd_NN)
    
    # Save model parameters
    #torch.save(network.state_dict(), 'PINN_state_03.10.23.pth')
    
