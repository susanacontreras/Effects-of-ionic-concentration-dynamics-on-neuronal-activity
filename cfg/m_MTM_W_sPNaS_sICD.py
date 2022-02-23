from scipy import integrate
from scipy.integrate import odeint
import numpy as np
from copy import copy

Pars_MTM_W_sPNaS_sICD={
    'I_app':1, #uA/cm2
    'I_app_m':1, #uA/cm2
    'I_app_a':1, #uA/cm2
    'cof':500,#Hz
    'C':1,#uF/cm2
    #Conductances
    'g_Na':100, #mS/cm2
    'g_K':200, #mS/cm2
    'g_L':0.1, #mS/cm2
    #Activation and inactivation parameters
    #I_Na
    'alpha_mV':54,
    'beta_mV':27,
    'alpha_hV':50,
    'beta_hV':27,
    # 'alpha_h_mag':0.128,
    # 'alpha_h_ts':18,
    #I_K
    'alpha_nV':52,
    'beta_nV':57,
    #Extracellular concentrations
    'Na_o':140,#mM
    'K_o':15,#mM
    #Cell properties
    'rho':4000,#check.. (cellular Area/Volume [1/cm]  Vitzthum,2009)
    #(http://nitrolab.engr.wisc.edu/pubs/vitzthumIB09.pdf)
    'remim':0.2,#Ratio ef extra to intra-cellular volume Zhang 2010,
    #( http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2920647/ )
    'P_K':0.96,
    'P_Na':0.04,
    'F':96484.6,#C/mol (Faraday constant)
    'R':8314.4,#mJ/[mol*K] (Ideal gas constant)
    'T':273.15+20,#K Temperature
    #Pump parameters
    'I_max':40,#uA/uF... Paper Luo y Rudi 1.5!!
    #Simple pump parameters
    'Na_sens0':20,
    'Na_sens':0.1,
    'parstNa':3,#Pump stoichiometry
    'parstK':2,#Pump stoichiometry
    #Relative Timing of ionic accumulation
    'r_fast_vs_ion':0.001,## Accounting for the fact the units of the change in concentrations is 1mM/S.. and the fast variables are in 1/ms
    #Temperature dependencies Q10s
    'T0':273.15+18,
    'Q10_gL':1.2,
    'Q10_gNa':1.2,
    'Q10_gK':1.2,
    'Q10_gPump':1.2,
    'Q10_n':2,
    'Q10_m':2,
    'Q10_h':2,
    #Initial conditions.. easy to manipulate from here at
    #the time of parameter exploration
    'm_Na0':0.1,
    'h_Na0':0.6,
    'n_K0':0.4,
    'Na_i0':10,
    'K_i0':150,#Very high...(To initialize it far from the dep block)
    'K_o0':8,
    'V0':-60
    }

# Mile traub model with simple pump[Na sensitive] definition
class neuron_MTM_W_sPNaS_sICD():

        def __init__(self, p):
            self.s_model_tag="Traub-Miles Model with Na sensitive Pump, and temperature dependencies"
            self.s_model_reference=""
            self.p = p
            self.n_state_vars=7
            self.s_state_vars=["m_Na", "h_Na", "n_K", "Na_i",
            "K_i", "K_o", "V"]
            self.s_currents=["I_Na", "I_K", "I_1", "i_p", "I_L","I_L_K", "I_L_Na"]
            self.s_concentrations=["Na_i","K_i","K_o"]
            self.s_resting_membrane_potentials=["E_Na","E_K","E_L"]
            self.step_protocol=[]
            self.current_state=[self.p['m_Na0'],self.p['h_Na0'],self.p['n_K0'],
            self.p['Na_i0'],self.p['K_i0'],self.p['K_o0'],self.p['V0']]
            self.ode_solver="odeint"
            self.noisy_current=[]
        def resting_membrane_potentials(self,v_State_Vars):
            #Resting potential for each neuron
            m_Na, h_Na, n_K, Na_i, K_i, K_o, V = v_State_Vars
            E_Na=self.p['R']*self.p['T']/self.p['F']*np.log(self.p['Na_o']/Na_i)
            E_K=self.p['R']*self.p['T']/self.p['F']*np.log(K_o/K_i)
            E_L=self.p['R']*self.p['T']/self.p['F']*np.log((K_o*self.p['P_K']+
            self.p['Na_o']*self.p['P_Na'])/(K_i*self.p['P_K']+Na_i*self.p['P_Na']))
            return E_Na, E_K, E_L

        def neuron_currents(self,v_State_Vars,t=[],i_p_f=[]):
            #Resting potential for each neuron
            m_Na, h_Na, n_K, Na_i, K_i, K_o, V = v_State_Vars# Voltage it's always the last state variable
            E_Na, E_K, E_L=self.resting_membrane_potentials(v_State_Vars)
            #Pump
            if i_p_f==[]:
                if Na_i<=self.p['Na_sens0']/2:
                    i_p=0
                if Na_i>self.p['Na_sens0']/2:
                    temp_factor_p=self.p['Q10_gPump']**(
                    (self.p['T']-self.p['T0'])/10)
                    i_p=self.p['I_max']*(1/(1+np.exp(-self.p['Na_sens']*(
                    Na_i-self.p['Na_sens0'])))-1/(1+np.exp(-self.p['Na_sens']*(
                    -self.p['Na_sens0']/2))))*temp_factor_p
            else:
                i_p=i_p_f(t) ##Argument to force the pump to be f_i_p value
            # Fast Na current
            I_Na=self.p['g_Na']*m_Na**3*h_Na*(V-E_Na)*self.p['Q10_gNa']**(
            (self.p['T']-self.p['T0'])/10)
            #K+ delayed rectifier
            I_K=self.p['g_K']*n_K**4*(V-E_K)*self.p['Q10_gK']**(
            (self.p['T']-self.p['T0'])/10)
            #Leak
            I_L_K=self.p['g_L']*self.p['P_K']*(V-E_K)*self.p['Q10_gL']**(
            (self.p['T']-self.p['T0'])/10)#[uA/cm2]
            I_L_Na=self.p['g_L']*self.p['P_Na']*(V-E_Na)*self.p['Q10_gL']**(
            (self.p['T']-self.p['T0'])/10)#[uA/cm2]
            I_L=I_L_K+I_L_Na
            I_1=I_L+i_p*(self.p['parstNa']-self.p['parstK'])#[uA/cm2]
            return I_Na, I_K, I_1, i_p, I_L, I_L_K, I_L_Na

        def activation_inactivation_Dynamics(self,v_State_Vars):
            m_Na, h_Na, n_K, Na_i, K_i, K_o, V = v_State_Vars
            #Activation and inactivation variables
            #fast Na current
            alpha_m=0.32*(V+self.p['alpha_mV'])/(1-np.exp(
            -(V+self.p['alpha_mV'])/4))*self.p['Q10_m']**(
            (self.p['T']-self.p['T0'])/10)
            beta_m=0.28*(V+self.p['beta_mV'])/(np.exp(
            (V+self.p['beta_mV'])/5)-1)*self.p['Q10_m']**(
            (self.p['T']-self.p['T0'])/10)
            alpha_h=0.128*np.exp(-(V+self.p['alpha_hV'])/18)*self.p['Q10_h']**(
            (self.p['T']-self.p['T0'])/10)
            beta_h=4/(1+np.exp(-(V+self.p['beta_hV'])/5))*self.p['Q10_h']**(
            (self.p['T']-self.p['T0'])/10)
            #K+ delayed rectifier
            alpha_n=0.032*(V+self.p['alpha_nV'])/(1
            -np.exp(-(V+self.p['alpha_nV'])/5))*self.p['Q10_n']**(
            (self.p['T']-self.p['T0'])/10)
            beta_n=0.5*np.exp(-(V+self.p['beta_nV'])/40)*self.p['Q10_n']**(
            (self.p['T']-self.p['T0'])/10)
            return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

        def neuron_changing_state(self,v_State_Vars,t,I_app,i_p_f):
            m_Na, h_Na, n_K, Na_i, K_i, K_o, V = v_State_Vars
            I_Na, I_K, I_1, i_p, I_L, I_L_K, I_L_Na=self.neuron_currents(v_State_Vars,t,i_p_f)
            alpha_m,beta_m,alpha_h,beta_h,alpha_n,beta_n=self.activation_inactivation_Dynamics(v_State_Vars)
            #Change of state variables
            dm_Na=alpha_m*(1-m_Na)-beta_m*m_Na
            dh_Na=alpha_h*(1-h_Na)-beta_h*h_Na
            dn_K=alpha_n*(1-n_K)-beta_n*n_K
            dNa_i=(self.p['rho']/self.p['F']*(
            -self.p['parstNa']*i_p-I_Na-I_L_Na)
            )*self.p['r_fast_vs_ion']
            dK_i=(self.p['rho']/self.p['F']*(self.p['parstK']*i_p
            -I_K-I_L_K))*self.p['r_fast_vs_ion']#Shut off o on..
            dK_o=-(dK_i)*self.p['remim']#Shut off o on..
            dV=1/self.p['C']*(I_app(t)-I_1-I_Na-I_K)
            return dm_Na, dh_Na, dn_K, dNa_i, dK_i, dK_o, dV

        def stimulate_neuron(self,t,c_ini,I_stim,i_p_f=[]):
            import copy
            sol= odeint(self.neuron_changing_state, c_ini,
            t, args=(I_stim,i_p_f))
            s_results=[]
            s_results=copy.copy(self.s_state_vars)
            s_results.append("t")
            t_prov=t[...,None]
            v_results=np.append(sol,t_prov,1)
            return s_results, v_results
