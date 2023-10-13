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
        
        # Plotting results from the ODE-Solver
        if plot:
            generate_data.plot(self, reactor_lengths, y[:,6])
            
        return y
    
    def plot(self, reactor_lengths, temperature):
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
        plt.plot(reactor_lengths, temperature, 'r-', label='temperature')
        plt.xlabel('reactor length')
        plt.ylabel('temperature')
        
    
        
if __name__ == "__main__":
    # Define parameters for the model
    reactor_lengths = np.linspace(0,12,num=1000)
    inlet_mole_fractions = [0.2128,0.714,0.0259,0.0004,0.0119,0.035] #CH4,H20,H2,CO,CO2,N2
    bound_conds = [25.7,2.14,793,1100] #p,u,T_in,T_wall
    reactor_conds = [0.007] #eta
    
    plot_analytical_solution = False #True,False
    
    # Calculation of the analytical curves
    model = generate_data(inlet_mole_fractions, bound_conds, reactor_conds)
    model.solve_ode(reactor_lengths, plot=plot_analytical_solution)
    