"""
Calculation of an industrial steam reformer considering the steam reforming 
reaction, water gas shift reaction and the direct reforming reaction. The 
reactor model corresponds to the final project from the Matlab course.
   
Code written by Alexander Keßler on 11.10.23
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
    def __init__(self, inlet_mole_fractions, bound_conds, reactor_conds):
        """Constructor:
        Args:
            inlet_mole_fractions (list): Inlet mole fractions of CH4, H20, H2, 
                                         CO, CO2 and N2 [-]
            bound_conds (list): Pressure [bar], flow velocity [m s-1], inlet 
                                temperature [K], temperature of the steam 
                                reformer wall [K]
            reactor_conds (list): efficiency factor of the catalyst [-]
            
        Params:
            x_CH4_0 (float): Inlet mole fraction of methane [-]
            x_H20_0 (float): Inlet mole fraction of water [-]
            x_H2_0 (float): Inlet mole fraction of hydrogen [-]
            x_CO_0 (float): Inlet mole fraction of carbon monoxide [-]
            x_CO2_0 (float): Inlet mole fraction of carbon dioxide [-]
            x_N2_0 (float): Inlet mole fraction of nitrogen [-]
            
            p (float): Pressure [bar]
            u (float): flow velocity [m s-1]
            T0 (int): inlet temperature [K]
            T_wall (int): temperature of the steam reformer wall [K]
            
            eta (float): efficiency factor of the catalyst [-]
            
            k0 (1D-array): preexponential factor/ arrhenius parameter [kmol Pa0.5 kgcat-1 h-1]
            E_A (1D-array): activation energy/ arrhenius parameter [J/mol]
            K_ads_0 (1D-array): adsorption constant [Pa-1] for K_ads_0[0:3]
            G_R_0 (1D-array): adsorption enthalpy [J mol-1]
            R (float): gas constant [J K-1 mol-1]
            A (float): cross section tube [m2]
            rho_b (float): density fixed bed [kg m-3]
            V_dot (float): volumetric flow rate [m^3 h-1]
            cp_coef (2D-array): coefficients of the NASA polynomials [j mol-1 K-1]
            H_0 (1D-array): enthalpies of formation [J mol-1]
            S_0 (1D-array): entropy of formation [J mol-1 K-1]
            MW (1D-array): molecular weight [kg mol-1]
            U_perV (float): thermal transmittance [kJ m-3 h-1 K-1]
        """
        
        self.x_CH4_0 = inlet_mole_fractions[0]
        self.x_H2O_0 = inlet_mole_fractions[1]
        self.x_H2_0 = inlet_mole_fractions[2]
        self.x_CO_0 = inlet_mole_fractions[3]
        self.x_CO2_0 = inlet_mole_fractions[4]
        self.x_N2_0 = inlet_mole_fractions[5]
        self.inlet_mole_fractions = np.array(inlet_mole_fractions[0:6])
        
        self.p = bound_conds[0]
        self.u = bound_conds[1]
        self.T0 = bound_conds[2]
        self.T_wall = bound_conds[3]
        
        self.eta = reactor_conds[0]
        
        self.k0 = np.array([4.225*1e15*math.sqrt(1e5),1.955*1e6*1e-5,1.020*1e15*math.sqrt(1e5)])
        self.E_A = np.array([240.1*1e3,67.13*1e3,243.9*1e3])
        self.K_ads_0 = np.array([8.23*1e-5*1e-5,6.12*1e-9*1e-5,6.65*1e-4*1e-5,1.77*1e5])
        self.G_R_ads = np.array([-70.61*1e3,-82.90*1e3,-38.28*1e3,88.68*1e3])
        self.R = 8.314472
        self.A = 0.0081
        self.rho_b = 1.3966*1e3
        self.V_dot = self.u * 3600 * self.A
        
        self.cp_coef = np.array([[19.238,52.09*1e-3,11.966*1e-6,-11.309*1e-9], \
                                 [32.22,1.9225*1e-3,10.548*1e-6,-3.594*1e-9], \
                                 [27.124,9.267*1e-3,-13.799*1e-6,7.64*1e-9], \
                                 [30.848,-12.84*1e-3,27.87*1e-6,-12.71*1e-9], \
                                 [19.78,73.39*1e-3,-55.98*1e-6,17.14*1e-9], \
                                 [31.128, -13.556*1e-3, 26.777*1e-6, -11.673*1e-9]])
        self.H_0 = np.array([-74850, -241820, 0, -110540, -393500])
        self.S_0 = np.array([-80.5467, -44.3736, 0,	89.6529, 2.9515])
        self.MW = np.array([16.043, 18.02, 2.016, 28.01, 44.01, 28.01]) * 1e-3
        self.U_perV = 2.3e+05
        
    def calc_inlet_ammount_of_substances(self):
        """
        Calculate the ammount of substances at the inlet of the PFR.
        
        New Params:
            total_concentration (float): Concentration of the gas [kmol m-3]
            concentrations (1D-array): Concentrations of all species [kmol m-3]
            ammount_of_substances (1D-array): Ammount of substances of all species [kmol h-1]
        """
        total_concentration = (self.p * 1e5 * 1e-3) / (self.R*self.T0)
        concentrations = total_concentration * np.array([self.x_CH4_0, self.x_H2O_0, \
                                                         self.x_H2_0, self.x_CO_0, \
                                                         self.x_CO2_0, self.x_N2_0])
        ammount_of_substances = concentrations * self.V_dot
        
        self.n_CH4_0 = ammount_of_substances[0]
        self.n_H2O_0 = ammount_of_substances[1]
        self.n_H2_0 = ammount_of_substances[2]
        self.n_CO_0 = ammount_of_substances[3]
        self.n_CO2_0 = ammount_of_substances[4]
        self.n_N2_0 = ammount_of_substances[5]
           
    def calc_mole_fractions(self, n_vector):
        """
        Calculate the mole fractions of all species.
        
        Args:
            n_vector (1D-array): Ammount of substances of all species [kmol h-1]
        """
        
        x_CH4 = n_vector[0]/np.sum(n_vector)
        x_H2O = n_vector[1]/np.sum(n_vector)
        x_H2 = n_vector[2]/np.sum(n_vector)
        x_CO = n_vector[3]/np.sum(n_vector)
        x_CO2 = n_vector[4]/np.sum(n_vector)
        x_N2 = n_vector[5]/np.sum(n_vector)
        
        self.mole_fractions = np.array([x_CH4, x_H2O, x_H2, x_CO, x_CO2, x_N2])
    
    def calc_mole_fractions_results(self, n_matrix):
        """
        Calculate the mole fractions of all species.
        
        Args:
            n_matrix (2D-array): Ammount of substances of all species depending 
                                 of the reactor length [kmol h-1]
        """
        
        self.x_CH4 = n_matrix[:,0]/np.sum(n_matrix, axis=1)
        self.x_H2O = n_matrix[:,1]/np.sum(n_matrix, axis=1)
        self.x_H2 = n_matrix[:,2]/np.sum(n_matrix, axis=1)
        self.x_CO = n_matrix[:,3]/np.sum(n_matrix, axis=1)
        self.x_CO2 = n_matrix[:,4]/np.sum(n_matrix, axis=1)
        self.x_N2 = n_matrix[:,5]/np.sum(n_matrix, axis=1)
    
    def calculate_thermo_properties(self):
        """
        Calculation of the reaction enthalpies and the equilibrium constants Kp.
        The temperature dependence of the heat capacity is considered with NASA 
        polynomials in order to consider the temperature dependence of the 
        enthalpies of formation and the entropies of formation. 
        
        New Params:
            H_i (1D-array): enthalpies of formation [J mol-1]
            S_i (1D-array): entropies of formation [J mol-1 K-1]
            H_R (1D-array): reaction enthalpies [J mol-1]
            S_R (1D-array): reaction entropies [J mol-1 K-1]
            G_R (1D-array): reaction gibbs energies [J mol-1]
            Kp (1D-array): equilibrium constant, Kp[1,3]=[Pa], Kp[2]=[-]
        """
        
        # Calculation of the standard formation enthalpy at given temperature
        # with Kirchhoffsches law
        H_i = self.H_0.reshape(-1,1) + np.matmul(self.cp_coef[:5,:], \
                np.array([[self.T], [(1/2)*self.T**2], [(1/3)*self.T**3], \
                [(1/4)*(self.T)**4]])) - np.matmul(self.cp_coef[:5,:], np.array( \
                [[298.15], [(1/2)*298.15**2], [(1/3)*298.15**3], [(1/4)*298.15**4]]))
        
        # Calculation of the standard formation entropy at given temperature
        S_i = self.S_0.reshape(-1,1) + np.matmul(self.cp_coef[:5,:], \
                np.array([[math.log(self.T)], [self.T], [(1/2)*self.T**2], \
                [(1/3)*self.T**3]])) - np.matmul(self.cp_coef[:5,:], \
                np.array([[math.log(298.15)], [298.15], [(1/2)*298.15**2], \
                          [(1/3)*298.15**3]]))
        
        # Calculation of standard reaction enthalpies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        H_R = np.zeros(3)
        H_R[0] = -H_i[0] - H_i[1] + H_i[3] + 3*H_i[2]
        H_R[1] = -H_i[3] - H_i[1] + H_i[4] + H_i[2]
        H_R[2] = -H_i[0] - 2*H_i[1] + H_i[4] + 4*H_i[2]
        
        # Calculation of standard reaction entropies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        S_R = np.zeros(3)
        S_R[0] = -S_i[0] - S_i[1] + S_i[3] + 3*S_i[2]
        S_R[1] = -S_i[3] - S_i[1] + S_i[4] + S_i[2]
        S_R[2] = -S_i[0] - 2*S_i[1] + S_i[4] + 4*S_i[2]
        
        # Calculation of the free reaction enthalpy with the Gibbs Helmoltz equation
        G_R = H_R - self.T * S_R
        
        # Calculation of the rate constants
        Kp = np.exp(-G_R / (self.R*self.T)) * np.array([1e10,1,1e10])
        
        return [Kp, H_R]
                                                   

    def xu_froment(self, partial_pressures):
        """
        Calculation of the differentials from the mass balance with the kinetic 
        approach of Xu, Froment.
        
        Args:
            partial_pressures (1D-array): partial pressures [Pa]
        New Params:
            k (list): velocity constant, k[1,3]=[kmol Pa0.5 kgcat-1 h-1]
                                         k[2]=[kmol Pa-1 kgcat-1 h-1]
            K_ads (1D-array): adsorption constants, K_ads[1-3]=[Pa-1]
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            dn_dz (1D-array): derivatives of the amounts of substances of the 
                              species in dependence of the reactor length [kmol m-1 h-1]
        """
        
        # Calculate reaction rates with a Langmuir-Hinshelwood-Houghen-Watson approach
        k = self.k0 * np.exp(-self.E_A/(self.R*self.T))
        K_ads = self.K_ads_0 * np.exp(-self.G_R_ads/(self.R*self.T))
        Kp, H_R = generate_data.calculate_thermo_properties(self)
        
        DEN = 1 + partial_pressures[3]*K_ads[0] + partial_pressures[2]*K_ads[1] + \
            partial_pressures[0]*K_ads[2] + (partial_pressures[1]*K_ads[3]) / partial_pressures[2]
        r_total = np.zeros(3)
        r_total[0] = (k[0] / (partial_pressures[2]**2.5)) * (partial_pressures[0] * \
                    partial_pressures[1] - (((partial_pressures[2]**3) * partial_pressures[3]) / \
                    Kp[0])) / (DEN**2)
        r_total[1] = (k[1] / partial_pressures[2]) * (partial_pressures[3] * \
                    partial_pressures[1] - ((partial_pressures[2] * partial_pressures[4]) / \
                    Kp[1])) / (DEN**2)
        r_total[2] = (k[2] / (partial_pressures[2]**3.5)) * (partial_pressures[0] * \
                    (partial_pressures[1]**2) - (((partial_pressures[2]**4) * partial_pressures[4]) / \
                    Kp[2])) / (DEN**2)
        
        # Calculate derivatives for the mass balance
        dn_dz = np.zeros(6)
        dn_dz[0] = self.eta * self.A * (-r_total[0] - r_total[2]) * self.rho_b
        dn_dz[1] = self.eta * self.A * (-r_total[0] - r_total[1] - 2*r_total[2]) * self.rho_b
        dn_dz[2] = self.eta * self.A * (3*r_total[0] + r_total[1] + 4*r_total[2]) * self.rho_b
        dn_dz[3] = self.eta * self.A * (r_total[0] - r_total[1]) * self.rho_b
        dn_dz[4] = self.eta * self.A * (r_total[1] + r_total[2]) * self.rho_b
        dn_dz[5] = 0

        return [dn_dz, r_total]
    
    def heat_balance(self, r_total, u_gas):
        """
        Calculation of the derivatives of the temperature according to the 
        reactor length.

        Args:
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            u_gas (float): flow velocity of the gas [m s-1]
            
        New Params:
            s_H (1D-array): heat production rate through the reaction [J m-3 h-1]
            density_gas (float): density of the gas [kg m-3]
            cp (1D-array): heat capacities of the species [J mol-1 K-1]
            s_H_ext (float): heat exchange rate with environment [J m-3 h-1]
            dTdz (float): derivatives of the temperature in dependence of the 
                          reactor length [K m-1]
        """
        
        ## Heat balance
        # Calculation of the source term for the reaction
        Kp, H_R = generate_data.calculate_thermo_properties(self)
        s_H = -(H_R * 1e3) * self.eta * r_total * self.rho_b
        
        # Calculation of the source term for external heat exchange
        density_gas = np.sum(self.p * 1e5 * self.mole_fractions * self.MW) / (self.R * self.T)
        cp = self.cp_coef[:,0] + self.cp_coef[:,1] * self.T + self.cp_coef[:,2] * \
            self.T**2 + self.cp_coef[:,3] * self.T**3
        s_H_ext = -self.U_perV * 1e3 * (self.T - self.T_wall)
        
        # Calculate derivative for the heat balance
        dTdz = (np.sum(s_H)+s_H_ext) / (u_gas * 3.6 * np.matmul(self.mole_fractions,(cp/self.MW)) * density_gas * 1e3)
        
        return dTdz
    
    def ODEs(y,z,self):
        """
        Function for the ODE solver.

        New Params:
            ammount_of_substances: ammount of substances of all species [kmol h-1]
            u_gas (float): flow velocity of the gas [m s-1]
            partial_pressures (1D-array): partial pressures [Pa]
        """
        
        # Extract the values
        ammount_of_substances = np.array(y[:6])
        self.T = y[6]
        
        # Calculate mole fractions
        generate_data.calc_mole_fractions(self,ammount_of_substances)
        
        # Consider dependence of temperature and gas composition of the flow velocity
        u_gas = self.u * (self.T/self.T0) * (np.sum(self.inlet_mole_fractions*self.MW) / \
                                             np.sum(self.mole_fractions*self.MW))
        
        # Calculate inlet partial pressures
        partial_pressures = self.p * 1e5 * self.mole_fractions
        
        ## Mass balance
        # Calculate reaction rates with a Langmuir-Hinshelwood-Houghen-Watson approach
        dn_dz, r_total = generate_data.xu_froment(self, partial_pressures)
        
        ## Heat balance
        dTdz = generate_data.heat_balance(self, r_total, u_gas)
        
        # Combine derivatives
        dydz = np.append(dn_dz, dTdz)

        return dydz
        
    def solve_ode(self, reactor_lengths, plot):
        """
        Solution of the ODEs for the mass balance and heat balance with the 
        scipy solver for the steam reformer. The results can then be plotted.
        
        Args:
            reactor_lengths (1D-array): reactor lengths for the ODE-Solver [m]
            plot (bool): plotting the results
        Params:
            y (2D-array): ammount of substances [kmol h-1], temperature [K]
        """
        
        # Calculate inlet ammount of substances
        generate_data.calc_inlet_ammount_of_substances(self)
        
        # Solve ODE for isotherm, adiabatic or polytrop reactor
        y = odeint(generate_data.ODEs, [self.n_CH4_0, self.n_H2O_0, self.n_H2_0, \
                                        self.n_CO_0, self.n_CO2_0, self.n_N2_0, \
                                        self.T0], reactor_lengths, args=(self,))
        
        # Calculate mole fractions
        generate_data.calc_mole_fractions_results(self, y[:,:6])
        
        # Store temperature
        self.T = y[:,6]
        
        # Plotting results from the ODE-Solver
        if plot:
            generate_data.plot(self, reactor_lengths)
            
        return y
    
    def plot(self, reactor_lengths):
        """
        Plotting results from the ODE-Solver.
        """
        
        # Mole fractions plot
        plt.figure()
        plt.plot(reactor_lengths, self.x_CH4, '-', label=r'$x_{\rm{CH_{4}}}$')
        plt.plot(reactor_lengths, self.x_H2O, '-', label=r'$x_{\rm{H_{2}O}}$')
        plt.plot(reactor_lengths, self.x_H2, '-', label=r'$x_{\rm{H_{2}}}$')
        plt.plot(reactor_lengths, self.x_CO, '-', label=r'$x_{\rm{CO}}$')
        plt.plot(reactor_lengths, self.x_CO2, '-', label=r'$x_{\rm{CO_{2}}}$')
        plt.xlabel('reactor length')
        plt.ylabel('mole fractions')
        plt.legend(loc='center right')
        
        # Temperature plot
        plt.figure()
        plt.plot(reactor_lengths, self.T, 'r-', label='temperature')
        plt.xlabel('reactor length')
        plt.ylabel('temperature')

        
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
        # The exponential function is used to ensure that the quantities 
        # are always positive. The multiplication of T0 provides a scaling of 
        # the high temperatures to a value close to 1 -> T/T0.
        x[:,0] = torch.exp(x[:,0])
        x[:,1] = torch.exp(x[:,1])
        x[:,2] = torch.exp(x[:,2])
        x[:,3] = torch.exp(x[:,3])
        x[:,4] = torch.exp(x[:,4])
        x[:,5] = torch.exp(x[:,5]) * self.T0
        
        return x

class PINN_loss(torch.nn.Module):
    def __init__(self, weight_factors, epsilon, inlet_mole_fractions, \
                 bound_conds, reactor_conds):
        super(PINN_loss, self).__init__()
        
        # New parameter
        self.w_n, self.w_T, self.w_GE_n, self.w_GE_T, self.w_IC_n, \
            self.w_IC_T = weight_factors
        self.epsilon = torch.tensor(epsilon)
        
        # Paramter known from the class generate_data()
        self.inlet_mole_fractions = torch.tensor(inlet_mole_fractions[0:6])
        
        self.p = torch.tensor(bound_conds[0])
        self.u = torch.tensor(bound_conds[1])
        self.T0 = torch.tensor(bound_conds[2])
        self.T_wall = torch.tensor(bound_conds[3])
        
        self.eta = torch.tensor(reactor_conds[0])
        
        self.k0 = torch.tensor([4.225*1e15*math.sqrt(1e5),1.955*1e6*1e-5,1.020*1e15*math.sqrt(1e5)])
        self.E_A = torch.tensor([240.1*1e3,67.13*1e3,243.9*1e3])
        self.K_ads_0 = torch.tensor([8.23*1e-5*1e-5,6.12*1e-9*1e-5,6.65*1e-4*1e-5,1.77*1e5])
        self.G_R_ads = torch.tensor([-70.61*1e3,-82.90*1e3,-38.28*1e3,88.68*1e3])
        self.R = torch.tensor(8.314472)
        self.A = torch.tensor(0.0081)
        self.rho_b = torch.tensor(1.3966*1e3)
        self.V_dot = self.u * torch.tensor(3600) * self.A
        
        self.cp_coef = torch.tensor([[19.238,52.09*1e-3,11.966*1e-6,-11.309*1e-9], \
                                     [32.22,1.9225*1e-3,10.548*1e-6,-3.594*1e-9], \
                                     [27.124,9.267*1e-3,-13.799*1e-6,7.64*1e-9], \
                                     [30.848,-12.84*1e-3,27.87*1e-6,-12.71*1e-9], \
                                     [19.78,73.39*1e-3,-55.98*1e-6,17.14*1e-9], \
                                     [31.128, -13.556*1e-3, 26.777*1e-6, -11.673*1e-9]])
        self.H_0 = torch.tensor([-74850, -241820, 0, -110540, -393500])
        self.S_0 = torch.tensor([-80.5467, -44.3736, 0,	89.6529, 2.9515])
        self.MW = torch.tensor([16.043, 18.02, 2.016, 28.01, 44.01, 28.01]) * 1e-3
        self.U_perV = torch.tensor(2.3e+05)
    
    def calc_inlet_ammount_of_substances(self):
        """
        Calculate the ammount of substances at the inlet of the PFR.
        
        New Params:
            total_concentration (float): Concentration of the gas [kmol m-3]
            concentrations (1D-array): Concentrations of all species [kmol m-3]
            ammount_of_substances (1D-array): Ammount of substances of all species [kmol h-1]
        """
        total_concentration = (self.p * 1e5 * 1e-3) / (self.R*self.T0)
        concentrations = total_concentration * self.inlet_mole_fractions
        ammount_of_substances = concentrations * self.V_dot
        
        self.n_CH4_0 = ammount_of_substances[0]
        self.n_H2O_0 = ammount_of_substances[1]
        self.n_H2_0 = ammount_of_substances[2]
        self.n_CO_0 = ammount_of_substances[3]
        self.n_CO2_0 = ammount_of_substances[4]
        self.n_N2_0 = ammount_of_substances[5]
    
    def calculate_thermo_properties(self):
        """
        Calculation of the reaction enthalpies and the equilibrium constants Kp.
        The temperature dependence of the heat capacity is considered with NASA 
        polynomials in order to consider the temperature dependence of the 
        enthalpies of formation and the entropies of formation. 
        
        New Params:
            H_i (1D-array): enthalpies of formation [J mol-1]
            S_i (1D-array): entropies of formation [J mol-1 K-1]
            H_R (1D-array): reaction enthalpies [J mol-1]
            S_R (1D-array): reaction entropies [J mol-1 K-1]
            G_R (1D-array): reaction gibbs energies [J mol-1]
            Kp (1D-array): equilibrium constant, Kp[1,3]=[Pa], Kp[2]=[-]
        """
        
        # Calculation of the standard formation enthalpy at given temperature
        # with Kirchhoffsches law
        H_i = torch.unsqueeze(self.H_0, 1).repeat(1, self.T.size(0)) + \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [self.T, (1/2)*self.T**2, (1/3)*self.T**3, (1/4)*(self.T)**4]], dim=0)) - \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [torch.full((self.T.size(0), 1), 298.15), (1/2)*torch.full((self.T.size(0), 1), 298.15)**2, \
                (1/3)*torch.full((self.T.size(0), 1), 298.15)**3, (1/4)*torch.full((self.T.size(0), 1), 298.15)**4]], dim=0))
            
        # Calculation of the standard formation entropy at given temperature
        S_i = torch.unsqueeze(self.S_0, 1).repeat(1, self.T.size(0)) + \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [torch.log(self.T), self.T, (1/2)*self.T**2, (1/3)*self.T**3]], dim=0)) - \
                torch.matmul(self.cp_coef[:5,:], torch.cat([torch.transpose(t, 0, 1) for t in \
                [torch.log(torch.full((self.T.size(0), 1), 298.15)), torch.full((self.T.size(0), 1), 298.15), \
                (1/2)*torch.full((self.T.size(0), 1), 298.15)**2, (1/3)*torch.full((self.T.size(0), 1), 298.15)**3]], dim=0))
                
        # Calculation of standard reaction enthalpies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        H_R = torch.zeros(3, 100)
        H_R[0,:] = -H_i[0,:] - H_i[1,:] + H_i[3,:] + 3*H_i[2,:]
        H_R[1,:] = -H_i[3,:] - H_i[1,:] + H_i[4,:] + H_i[2,:]
        H_R[2,:] = -H_i[0,:] - 2*H_i[1,:] + H_i[4,:] + 4*H_i[2,:]
        
        # Calculation of standard reaction entropies with Satz von Hess at 
        # standard conditions (T = 298.15 K)
        S_R = torch.zeros(3, 100)
        S_R[0,:] = -S_i[0,:] - S_i[1,:] + S_i[3,:] + 3*S_i[2,:]
        S_R[1,:] = -S_i[3,:] - S_i[1,:] + S_i[4,:] + S_i[2,:]
        S_R[2,:] = -S_i[0,:] - 2*S_i[1,:] + S_i[4,:] + 4*S_i[2,:]
        
        # Calculation of the free reaction enthalpy with the Gibbs Helmoltz equation
        G_R = H_R - self.T.t() * S_R
        
        # Calculation of the rate constants
        Kp = torch.exp(-G_R / (self.R*self.T.t())) * torch.cat((torch.full((1, 100), 1e10), \
                torch.full((1, 100), 1), torch.full((1, 100), 1e10)), dim=0)
        
        return [Kp.t(), H_R.t()]
                                                   

    def xu_froment(self, partial_pressures):
        """
        Calculation of the differentials from the mass balance with the kinetic 
        approach of Xu, Froment.
        
        Args:
            partial_pressures (1D-array): partial pressures [Pa]
        New Params:
            k (list): velocity constant, k[1,3]=[kmol Pa0.5 kgcat-1 h-1]
                                         k[2]=[kmol Pa-1 kgcat-1 h-1]
            K_ads (1D-array): adsorption constants, K_ads[1-3]=[Pa-1]
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            dn_dz (1D-array): derivatives of the amounts of substances of the 
                              species in dependence of the reactor length [kmol m-1 h-1]
        """
        
        # Calculate reaction rates with a Langmuir-Hinshelwood-Houghen-Watson approach
        k = self.k0.expand(self.T.size(0), -1) * torch.exp(-self.E_A.expand(self.T.size(0), -1)/(self.R*self.T))
        K_ads = self.K_ads_0.expand(self.T.size(0), -1) * torch.exp(-self.G_R_ads.expand(self.T.size(0), -1)/(self.R*self.T))
        Kp, H_R = PINN_loss.calculate_thermo_properties(self)
        
        DEN = 1 + partial_pressures[:,3]*K_ads[:,0] + partial_pressures[:,2]*K_ads[:,1] + \
            partial_pressures[:,0]*K_ads[:,2] + (partial_pressures[:,1]*K_ads[:,3]) / partial_pressures[:,2]
        r_total = torch.zeros(100, 3)
        r_total[:,0] = (k[:,0] / (partial_pressures[:,2]**2.5)) * (partial_pressures[:,0] * \
                        partial_pressures[:,1] - (((partial_pressures[:,2]**3) * partial_pressures[:,3]) / \
                        Kp[:,0])) / (DEN**2)
        r_total[:,1] = (k[:,1] / partial_pressures[:,2]) * (partial_pressures[:,3] * \
                        partial_pressures[:,1] - ((partial_pressures[:,2] * partial_pressures[:,4]) / \
                        Kp[:,1])) / (DEN**2)
        r_total[:,2] = (k[:,2] / (partial_pressures[:,2]**3.5)) * (partial_pressures[:,0] * \
                        (partial_pressures[:,1]**2) - (((partial_pressures[:,2]**4) * partial_pressures[:,4]) / \
                        Kp[:,2])) / (DEN**2)
        
        # Calculate derivatives for the mass balance
        dn_dz = torch.zeros(100, 5)
        dn_dz[:,0] = self.eta * self.A * (-r_total[:,0] - r_total[:,2]) * self.rho_b
        dn_dz[:,1] = self.eta * self.A * (-r_total[:,0] - r_total[:,1] - 2*r_total[:,2]) * self.rho_b
        dn_dz[:,2] = self.eta * self.A * (3*r_total[:,0] + r_total[:,1] + 4*r_total[:,2]) * self.rho_b
        dn_dz[:,3] = self.eta * self.A * (r_total[:,0] - r_total[:,1]) * self.rho_b
        dn_dz[:,4] = self.eta * self.A * (r_total[:,1] + r_total[:,2]) * self.rho_b

        return [dn_dz, r_total]
    
    def heat_balance(self, r_total, u_gas):
        """
        Calculation of the derivatives of the temperature according to the 
        reactor length.

        Args:
            r_total (1D-array): reaction rates [kmol kgcat-1 h-1]
            u_gas (float): flow velocity of the gas [m s-1]
            
        New Params:
            s_H (1D-array): heat production rate through the reaction [J m-3 h-1]
            density_gas (float): density of the gas [kg m-3]
            cp (1D-array): heat capacities of the species [J mol-1 K-1]
            s_H_ext (float): heat exchange rate with environment [J m-3 h-1]
            dTdz (float): derivatives of the temperature in dependence of the 
                          reactor length [K m-1]
        """
        
        ## Heat balance
        # Calculation of the source term for the reaction
        Kp, H_R = PINN_loss.calculate_thermo_properties(self)
        s_H = -(H_R * 1e3) * self.eta * r_total * self.rho_b
        
        # Calculation of the source term for external heat exchange
        density_gas = torch.sum(self.p * 1e5 * self.mole_fractions * \
            self.MW.unsqueeze(0).expand(100, -1), dim=1).unsqueeze(1) / (self.R * self.T)
        cp = self.cp_coef[:,0].unsqueeze(0).expand(100, -1) + \
            self.cp_coef[:,1].unsqueeze(0).expand(100, -1) * self.T + \
            self.cp_coef[:,2].unsqueeze(0).expand(100, -1) * self.T**2 + \
            self.cp_coef[:,3].unsqueeze(0).expand(100, -1) * self.T**3
        s_H_ext = -self.U_perV * 1e3 * (self.T - self.T_wall)
        
        # Calculate derivative for the heat balance
        dT_dz = (torch.sum(s_H,dim=1)+s_H_ext.squeeze()) / (u_gas.squeeze() * 3.6 * \
            torch.sum(self.mole_fractions*(cp/self.MW),dim=1) * density_gas.squeeze(1) * 1e3)
        
        return dT_dz
    
    def calc_IC_loss(self, y_pred, x):
        
        # Calculation of the mean square displacement between the predicted and 
        # original initial conditions
        loss_IC_CH4 = (y_pred[0, 0] - self.n_CH4_0)**2
        loss_IC_H2O = (y_pred[0, 1] - self.n_H2O_0)**2
        loss_IC_H2 = (y_pred[0, 2] - self.n_H2_0)**2
        loss_IC_CO = (y_pred[0, 3] - self.n_CO_0)**2
        loss_IC_CO2 = (y_pred[0, 4] - self.n_CO2_0)**2
        loss_IC_T = (y_pred[0, 5] - self.T0)**2
        
        return [loss_IC_CH4, loss_IC_H2O, loss_IC_H2, loss_IC_CO, loss_IC_CO2, loss_IC_T]
    
    def calc_GE_loss(self, y_pred, x):
           
        # Calculate the gradients of tensor values
        dn_dz_CH4 = torch.autograd.grad(outputs=y_pred[:, 0], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 0]),
                                retain_graph=True, create_graph=True)[0]
        dn_dz_H2O = torch.autograd.grad(outputs=y_pred[:, 1], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 1]),
                                retain_graph=True, create_graph=True)[0]     
        dn_dz_H2 = torch.autograd.grad(outputs=y_pred[:, 2], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 2]),
                                retain_graph=True, create_graph=True)[0]  
        dn_dz_CO = torch.autograd.grad(outputs=y_pred[:, 3], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 3]),
                                retain_graph=True, create_graph=True)[0]  
        dn_dz_CO2 = torch.autograd.grad(outputs=y_pred[:, 4], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 4]),
                                retain_graph=True, create_graph=True)[0]  
        dT_dz = torch.autograd.grad(outputs=y_pred[:, 5], inputs=x,
                                grad_outputs=torch.ones_like(y_pred[:, 5]),
                                retain_graph=True, create_graph=True)[0]
        
        # Calculation of the differentials
        # Calculate the mole fractions
        self.mole_fractions = torch.cat([y_pred[:,:5], model.n_N2_0*torch.ones(100, 1)], dim=1)/ \
            torch.sum(torch.cat([y_pred[:,:5], model.n_N2_0*torch.ones(100, 1)], dim=1), dim=1).view(-1, 1)
        
        #### Vorsicht! rausnehmen
        #self.mole_fractions[:,0] = 2.1280e-01
        #self.mole_fractions[:,1] = 7.1400e-01
        #self.mole_fractions[:,2] = 2.5900e-02
        #self.mole_fractions[:,3] = 4.0000e-04
        #self.mole_fractions[:,4] = 1.1900e-02
        #self.mole_fractions[:,5] = 0.035
        
        #### Vorsicht! rausnehmen
        #self.T.fill_(self.T0)
        
        # Consider dependence of temperature and gas composition of the flow velocity
        u_gas = self.u * (self.T/self.T0.expand(self.T.size(0), 1)) * (torch.sum(torch.unsqueeze( \
                    self.inlet_mole_fractions,dim=0).expand(self.T.size(0), -1)*self.MW, dim=1) / \
                    torch.sum(self.mole_fractions*self.MW, dim=1)).unsqueeze(1)
        
        # Calculate partial pressures
        partial_pressures = self.p * 1e5 * self.mole_fractions
        
        #### Vorsicht! rausnehmen
        #partial_pressures[:,0] = 5.46896e+05
        #partial_pressures[:,1] = 1.83498e+06
        #partial_pressures[:,2] = 6.65630e+04
        #partial_pressures[:,3] = 1.02800e+03
        #partial_pressures[:,4] = 3.05830e+04
        #u_gas = 2.1400
        
        dn_dz_pred, r_total_pred = PINN_loss.xu_froment(self, partial_pressures)
        dT_dz_pred = PINN_loss.heat_balance(self, r_total_pred, u_gas)
        
        # Calculation of the mean square displacement between the gradients of 
        # autograd and differentials of the mass and heat balances
        loss_GE_CH4 = (dn_dz_CH4 - dn_dz_pred[:,0].unsqueeze(1))**2
        loss_GE_H2O = (dn_dz_H2O - dn_dz_pred[:,1].unsqueeze(1))**2
        loss_GE_H2 = (dn_dz_H2 - dn_dz_pred[:,2].unsqueeze(1))**2
        loss_GE_CO = (dn_dz_CO - dn_dz_pred[:,3].unsqueeze(1))**2
        loss_GE_CO2 = (dn_dz_CO2 - dn_dz_pred[:,4].unsqueeze(1))**2
        loss_GE_T = (dT_dz - dT_dz_pred.unsqueeze(1))**2
        
        return [loss_GE_CH4, loss_GE_H2O, loss_GE_H2, loss_GE_CO, loss_GE_CO2, loss_GE_T]
    
    def causal_training(self, x, loss_IC_CH4, loss_IC_H2O, loss_IC_H2, \
                        loss_IC_CO, loss_IC_CO2, loss_IC_T, loss_GE_CH4, \
                        loss_GE_H2O, loss_GE_H2, loss_GE_CO, loss_GE_CO2, \
                        loss_GE_T):
        
        # Form the sum of the losses
        losses = torch.zeros_like(x)
        losses[0,0] = loss_IC_CH4 + loss_IC_H2O + loss_IC_H2 + loss_IC_CO + \
            loss_IC_CO2 + loss_IC_T + loss_GE_CH4[0] + loss_GE_H2O[0] + \
                loss_GE_H2[0] + loss_GE_CO[0] + loss_GE_CO2[0] + loss_GE_T[0]
        losses[1:] = loss_GE_CH4[1:] + loss_GE_H2O[1:] + loss_GE_H2[1:] + \
            loss_GE_CO[1:] + loss_GE_CO2[1:] + loss_GE_T[1:]
        
        # Calculate the weighting factors of the losses
        weight_factors = torch.zeros_like(x)
        for i in range(x.size(0)):
            w_i = torch.exp(-self.epsilon*torch.sum(losses[0:i]))
            weight_factors[i] = w_i
        self.weight_factors = weight_factors
        
        # Consider the weighting factors in the losses
        total_loss = torch.mean(weight_factors * losses)
            
        return total_loss
         
    def forward(self, x, y, y_pred):
        
        # Calculation of the total loss
        self.T = y_pred[:, 5].reshape(-1, 1)
        
        loss_IC_CH4, loss_IC_H2O, loss_IC_H2, loss_IC_CO, \
            loss_IC_CO2, loss_IC_T = self.calc_IC_loss(y_pred, x)
        loss_GE_CH4, loss_GE_H2O, loss_GE_H2, loss_GE_CO, \
            loss_GE_CO2, loss_GE_T = self.calc_GE_loss(y_pred, x)
        
        # Store losses before weighting
        losses_before_weighting = np.array([np.mean(loss_IC_CH4.detach().numpy()), \
                                   np.mean(loss_IC_H2O.detach().numpy()), \
                                   np.mean(loss_IC_H2.detach().numpy()), \
                                   np.mean(loss_IC_CO.detach().numpy()), \
                                   np.mean(loss_IC_CO2.detach().numpy()), \
                                   np.mean(loss_IC_T.detach().numpy()), \
                                   np.mean(loss_GE_CH4.detach().numpy()), \
                                   np.mean(loss_GE_H2O.detach().numpy()), \
                                   np.mean(loss_GE_H2.detach().numpy()), \
                                   np.mean(loss_GE_CO.detach().numpy()), \
                                   np.mean(loss_GE_CO2.detach().numpy()), \
                                   np.mean(loss_GE_T.detach().numpy())])

        # Consider weighting factors of the loss functions
        loss_IC_CH4 = self.w_IC_n * self.w_n * loss_IC_CH4
        loss_IC_H2O = self.w_IC_n * self.w_n * loss_IC_H2O
        loss_IC_H2 = self.w_IC_n * self.w_n * loss_IC_H2
        loss_IC_CO = self.w_IC_n * self.w_n * loss_IC_CO
        loss_IC_CO2 = self.w_IC_n * self.w_n * loss_IC_CO2
        loss_IC_T = self.w_IC_T * self.w_T * loss_IC_T
        loss_GE_CH4 = self.w_GE_n * self.w_n * loss_GE_CH4
        loss_GE_H2O = self.w_GE_n * self.w_n * loss_GE_H2O
        loss_GE_H2 = self.w_GE_n * self.w_n * loss_GE_H2
        loss_GE_CO = self.w_GE_n * self.w_n * loss_GE_CO
        loss_GE_CO2 = self.w_GE_n * self.w_n * loss_GE_CO2
        loss_GE_T = self.w_GE_T * self.w_T * loss_GE_T
        
        # Store losses after weighting
        losses_after_weighting = np.array([np.mean(loss_IC_CH4.detach().numpy()), \
                                   np.mean(loss_IC_H2O.detach().numpy()), \
                                   np.mean(loss_IC_H2.detach().numpy()), \
                                   np.mean(loss_IC_CO.detach().numpy()), \
                                   np.mean(loss_IC_CO2.detach().numpy()), \
                                   np.mean(loss_IC_T.detach().numpy()), \
                                   np.mean(loss_GE_CH4.detach().numpy()), \
                                   np.mean(loss_GE_H2O.detach().numpy()), \
                                   np.mean(loss_GE_H2.detach().numpy()), \
                                   np.mean(loss_GE_CO.detach().numpy()), \
                                   np.mean(loss_GE_CO2.detach().numpy()), \
                                   np.mean(loss_GE_T.detach().numpy())])
        
        # Calculate the total loss
        total_loss = PINN_loss.causal_training(self, x, loss_IC_CH4, loss_IC_H2O, \
                        loss_IC_H2, loss_IC_CO, loss_IC_CO2, loss_IC_T, loss_GE_CH4, \
                            loss_GE_H2O, loss_GE_H2, loss_GE_CO, loss_GE_CO2, loss_GE_T)
        
        return [total_loss, losses_before_weighting, losses_after_weighting]

def train(x, y, network, calc_loss, optimizer, num_epochs, analytical_solution_x_CH4, \
          analytical_solution_x_H20, analytical_solution_x_H2, analytical_solution_x_CO, \
          analytical_solution_x_CO2, analytical_solution_x_N2, analytical_solution_T, \
          n_N2_0, plot_interval):
    
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
            calc_loss(x, y, ypred)
        
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
    
    # Prepare folders for the plots 
    prepare_plots_folder()
    
    # Training loop
    calc_loss.calc_inlet_ammount_of_substances()
    
    loss_values = np.zeros((num_epochs,25))
    for epoch in range(num_epochs):
        # Updating the network parameters with calculated losses and gradients 
        # from the closure-function
        optimizer.step(closure)

        # Calculation of the current total loss after updating the network parameters 
        # in order to print them in the console afterwards and plot them.
        total_loss, losses_before_weighting, losses_after_weighting = \
            calc_loss(x, y, network(x))
        loss_values[epoch,:] = np.hstack((np.array(total_loss.detach().numpy()),\
                                          losses_before_weighting,losses_after_weighting))

        print('Epoch: ', epoch+1, 'Loss: ', total_loss.item(), 'causal weights sum: ', \
              np.round(np.sum(calc_loss.weight_factors.detach().numpy()),decimals=4))
        
        # Create plots in the given plot_interval
        if (epoch+1)%plot_interval == 0 or epoch == 0:
            y_pred = network(x)
            y_pred = y_pred.detach().numpy()
            y_pred = np.insert(y_pred, 2, n_N2_0, axis=1)
            predicted_x_CH4, predicted_x_H20, predicted_x_H2, predicted_x_CO, \
                predicted_x_CO2, predicted_x_N2 = calc_mole_frac(y_pred[:,:6])
            predicted_T = y_pred[:, 6]
            
            plots(x.detach().numpy(),analytical_solution_x_CH4, analytical_solution_x_H20, \
                  analytical_solution_x_H2, analytical_solution_x_CO, analytical_solution_x_CO2, \
                  analytical_solution_x_N2, analytical_solution_T, predicted_x_CH4, predicted_x_H20, \
                  predicted_x_H2, predicted_x_CO, predicted_x_CO2, predicted_x_N2, predicted_T, save_plot=True, \
                        plt_num=epoch+1, weight_factors=calc_loss.weight_factors)
            
    return [loss_values]    

def calc_mole_frac(n_matrix):
    """
    Calculate mole fractions from the ammount of substances. Takes into account 
    only two species.
    
    Args:
        n_matrix (2D-array): ammount of substance from species A and B in 
                             dependence of the reactor length
    """

    x_CH4 = n_matrix[:,0]/np.sum(n_matrix, axis=1)
    x_H20 = n_matrix[:,1]/np.sum(n_matrix, axis=1)
    x_H2 = n_matrix[:,2]/np.sum(n_matrix, axis=1)
    x_CO = n_matrix[:,3]/np.sum(n_matrix, axis=1)
    x_CO2 = n_matrix[:,4]/np.sum(n_matrix, axis=1)
    x_N2 = n_matrix[:,5]/np.sum(n_matrix, axis=1)
    
    return [x_CH4,x_H20,x_H2,x_CO,x_CO2,x_N2]

def plots(reactor_lengths, analytical_solution_x_CH4, analytical_solution_x_H20, \
      analytical_solution_x_H2, analytical_solution_x_CO, analytical_solution_x_CO2, \
      analytical_solution_x_N2, analytical_solution_T, predicted_x_CH4, predicted_x_H20, \
      predicted_x_H2, predicted_x_CO, predicted_x_CO2, predicted_x_N2, predicted_T, \
      save_plot=False, plt_num=None, weight_factors=None, loss_values=None, msd_NN=None):
    """
    This function is used to save the plots during the training and to plot 
    them after the training. 

    """
    # Path to the folder with the plots
    script_folder = os.path.dirname(os.path.abspath(__file__))
    folder_name = "plots"
    folder_path = os.path.join(script_folder, folder_name)
    
    # Mole fractions plot
    plt.figure()
    plt.plot(reactor_lengths, analytical_solution_x_CH4, 'g-', label=r'$x_{\rm{CH_{4}}}$')
    plt.scatter(reactor_lengths, predicted_x_CH4, color = 'g', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{CH_{4},pred}}$')
    plt.plot(reactor_lengths, analytical_solution_x_H20, 'r-', label=r'$x_{\rm{H_{2}O}}$')
    plt.scatter(reactor_lengths, predicted_x_H20, color = 'r', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{H_{2}O,pred}}$')
    plt.plot(reactor_lengths, analytical_solution_x_H2, 'm-', label=r'$x_{\rm{H_{2}}}$')
    plt.scatter(reactor_lengths, predicted_x_H2, color = 'm', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{H_{2},pred}}$')
    plt.plot(reactor_lengths, analytical_solution_x_CO, 'c-', label=r'$x_{\rm{CO}}$')
    plt.scatter(reactor_lengths, predicted_x_CO, color = 'c', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{CO,pred}}$')
    plt.plot(reactor_lengths, analytical_solution_x_CO2, 'y-', label=r'$x_{\rm{CO_{2}}}$')
    plt.scatter(reactor_lengths, predicted_x_CO2, color = 'y', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{CO_{2},pred}}$')
    plt.plot(reactor_lengths, analytical_solution_x_N2, 'b-', label=r'$x_{\rm{N_{2}}}$')
    plt.scatter(reactor_lengths, predicted_x_N2, color = 'b', s=12, alpha=0.8, \
                edgecolors='b', label=r'$x_{\rm{N_{2},pred}}$')
    plt.xlabel(r'$reactor\:length\:/\:\rm{m}$')
    plt.ylabel(r'$mole\:fractions$')
    plt.ylim(0,0.75)
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
    plt.ylabel(r'$temperature\:/\:\rm{T}$')
    plt.ylim(analytical_solution_T[0],analytical_solution_T[-1])
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
        epochs = list(range(1, len(loss_values[0])+1))
        plt.plot(epochs, loss_values[0][:,0], '-', label=r'$L_{\rm{total}}$')
        plt.plot(epochs, loss_values[0][:,1], '-', label=r'$L_{\rm{IC,CH_{4}}}$')
        plt.plot(epochs, loss_values[0][:,2], '-', label=r'$L_{\rm{IC,H_{2}O}}$')
        plt.plot(epochs, loss_values[0][:,3], '-', label=r'$L_{\rm{IC,H_{2}}}$')
        plt.plot(epochs, loss_values[0][:,4], '-', label=r'$L_{\rm{IC,CO}}$')
        plt.plot(epochs, loss_values[0][:,5], '-', label=r'$L_{\rm{IC,CO_{2}}}$')
        plt.plot(epochs, loss_values[0][:,6], '-', label=r'$L_{\rm{IC,T}}$')
        plt.plot(epochs, loss_values[0][:,7], '--', label=r'$L_{\rm{GE,CH_{4}}}$')
        plt.plot(epochs, loss_values[0][:,8], '--', label=r'$L_{\rm{GE,H_{2}O}}$')
        plt.plot(epochs, loss_values[0][:,9], '--', label=r'$L_{\rm{GE,H_{2}}}$')
        plt.plot(epochs, loss_values[0][:,10], '--', label=r'$L_{\rm{GE,CO}}$')
        plt.plot(epochs, loss_values[0][:,11], '--', label=r'$L_{\rm{GE,CO_{2}}}$')
        plt.plot(epochs, loss_values[0][:,12], '--', label=r'$L_{\rm{GE,T}}$')
        
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
    reactor_lengths = np.linspace(0,12,num=100)
    inlet_mole_fractions = [0.2128,0.714,0.0259,0.0004,0.0119,0.035] #CH4,H20,H2,CO,CO2,N2
    bound_conds = [25.7,2.14,793,1100] #p,u,T_in,T_wall
    reactor_conds = [0.007] #eta
    
    plot_analytical_solution = False #True,False
    
    input_size_NN = 1
    hidden_size_NN = 32
    output_size_NN = 6
    num_layers_NN = 3
    num_epochs = 300
    weight_factors = [1e2,1,1,1,1,1] #w_n,w_T,w_GE_n,w_GE_T,w_IC_n,w_IC_T
    epsilon = 0 #epsilon=0: old model, epsilon!=0: new model, optimized value: 2
    plot_interval = 10 # Plotting during NN-training
    
    
    # Calculation of the analytical curves
    model = generate_data(inlet_mole_fractions, bound_conds, reactor_conds)
    model.solve_ode(reactor_lengths, plot=plot_analytical_solution)
    
    analytical_solution_x_CH4 = model.x_CH4
    analytical_solution_x_H20 = model.x_H2O
    analytical_solution_x_H2 = model.x_H2
    analytical_solution_x_CO = model.x_CO
    analytical_solution_x_CO2 = model.x_CO2
    analytical_solution_x_N2 = model.x_N2
    analytical_solution_T = model.T
    
    # Set up the neural network
    network = NeuralNetwork(input_size_NN=input_size_NN, hidden_size_NN=hidden_size_NN,\
                            output_size_NN=output_size_NN, num_layers_NN=num_layers_NN,\
                            T0 = bound_conds[2])
    optimizer = torch.optim.LBFGS(network.parameters(), lr=1, line_search_fn= \
                                  "strong_wolfe", max_eval=None, tolerance_grad \
                                      =1e-50, tolerance_change=1e-50)
        
    x = torch.tensor(reactor_lengths.reshape(-1, 1), requires_grad=True)
    y = None
    
    # Train the neural network
    calc_loss = PINN_loss(weight_factors, epsilon, inlet_mole_fractions, \
                          bound_conds, reactor_conds) 
        
    loss_values = train(x, y, network, calc_loss, optimizer, num_epochs, analytical_solution_x_CH4, \
              analytical_solution_x_H20, analytical_solution_x_H2, analytical_solution_x_CO, \
              analytical_solution_x_CO2, analytical_solution_x_N2, analytical_solution_T, \
              model.n_N2_0, plot_interval)

    # Predict data with the trained neural network
    y_pred = network(x)
    # Convert the pytorch tensor to a numpy array
    y_pred = y_pred.detach().numpy()
    # Add the amount of substance from N2 to the the predicted data because N2 
    # is not taken into account in the neural network (N2 stays constant)
    y_pred = np.insert(y_pred, 2, model.n_N2_0, axis=1)
    # Extract mole fractions
    predicted_x_CH4, predicted_x_H20, predicted_x_H2, predicted_x_CO, \
        predicted_x_CO2, predicted_x_N2 = calc_mole_frac(y_pred[:,:6])
    predicted_T = y_pred[:, 6]
    
    # Plot the mole fraction, temperature and weighting factors    
    plots(reactor_lengths, analytical_solution_x_CH4, analytical_solution_x_H20, \
          analytical_solution_x_H2, analytical_solution_x_CO, analytical_solution_x_CO2, \
          analytical_solution_x_N2, analytical_solution_T, predicted_x_CH4, predicted_x_H20, \
          predicted_x_H2, predicted_x_CO, predicted_x_CO2, predicted_x_N2, predicted_T, \
          plt_num = f'{num_epochs}_final', save_plot = True, loss_values=loss_values)
    
    # Save model parameters
    #torch.save(network.state_dict(), 'PINN_state_03.10.23.pth')