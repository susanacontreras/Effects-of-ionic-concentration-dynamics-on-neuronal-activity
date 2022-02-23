import pylab, os, json, sympy, scipy, sys
from sympy import S, symbols, lambdify
from scipy import integrate
from scipy.integrate import odeint
import numpy as np
from copy import copy

def load_mod(modfile,bp):

    ### Loads expressions from the json file
    nakdic = json.load(open(modfile))
    ### Auxiliary functions..
    fundic = dict([(j,k.split(":")[0]) for j,k in nakdic["functions"].items()])
    ### Definitions..
    defdic = dict([(j,k.split(":")[0]) for j,k in nakdic["definitions"].items()])
    ### Parameters
    pardic = dict([(j,k) for j,k in nakdic["parameters"].items()])
    ### Initial conditions
    icdic0 =  [[j,k] for j,k in nakdic["init_states"].items()]
    icdic = [(i,str(S(str(S(str(S(str(S(j).subs(defdic))).subs(fundic))).subs(fundic))).subs(pardic))) for i,j in icdic0]
    ### parameters to be changed
    bifpar = [(k,pardic.pop(k)) for k in bp]
    ### Current expressions
    if 'currents' in nakdic.keys():
        currentlist1 =  [[j,k] for j,k in nakdic['currents'].items()]
        currentlist = [(i,str(S(str(S(str(S(str(S(j).subs(defdic))).subs(fundic))).subs(fundic))).subs(pardic))) for i,j in currentlist1]

    if 'currents' not in nakdic.keys():
        currentlist={}

    ### SS Current expressions
    if 'ss_currents' in nakdic.keys():
        ss_currents1=[[j,k] for j,k in nakdic['ss_currents'].items()]
        ss_currents = [(i,str(S(str(S(str(S(str(S(j).subs(defdic))).subs(fundic))).subs(fundic))).subs(pardic))) for i,j in ss_currents1]

    if 'ss_currents' not in nakdic.keys():
        ss_currents={}

    ### Membrane potential expressions
    if 'resting_membrane_pot' in nakdic.keys():
        membpotlist1 =  [[j,k] for j,k in nakdic['resting_membrane_pot'].items()]
        membpotlist = [(i,str(S(str(S(str(S(str(S(j).subs(defdic))).subs(fundic))).subs(fundic))).subs(pardic))) for i,j in membpotlist1]
    if 'resting_membrane_pot' not in nakdic.keys():
        membpotlist={}

    ### ODEs
    statevar_list = list(nakdic["init_states"].keys())
    time_derivative_list= ["d{}/dt".format(k) for k in statevar_list]
    sdelist=[]
    for ode in nakdic["ode"]:
      odestr= "({})-({})".format(*ode.split("="))
      odestr= odestr.replace(" ","")
      time_derivative= [k for k in time_derivative_list if k in odestr][0] # bad style
      state_variable= [k for k in statevar_list if k in time_derivative][0]
      ode_rhs= sympy.S(sympy.solve(odestr,time_derivative)[0]) # [0] is bad style
      sdelisti = [state_variable,str(S(str(S(str(S(str(S(ode_rhs).subs(defdic))).subs(fundic))).subs(fundic))).subs(pardic))]
      sdelist+=[sdelisti]
    ### Model Description
    if 'source' in nakdic.keys():
        s_description=nakdic['source']
    if 'source' not in nakdic.keys():
        s_description=''
        if "comments" in nakdic.keys():
            if "source" in nakdic['comments'].keys():
                s_description=nakdic['comments']['source']

    return sdelist, currentlist, membpotlist, s_description, icdic, pardic, ss_currents

class generic_neuron_from_json():

        def __init__(self, s_name_model,dir_file=[],strIapp='I_app'):
            ### Creates a class generic_neuron_from_json() which contains all the expressions that describe the dynamics of [s_name_model]
            ## s_name_model= file name efample: 'MTM_W_PNAs_Temp_snapshot_constleak.json'
            ## dir_file= path of the file containning cfg/
            # For practival reasons I_app is always a parameter.. (to be used for stimulationg with time dependent function)
            bifpar={}
            if bifpar=={}:
                I_app=-5
                bifpar = {
                  strIapp : [str(I_app)+"* uA/cm2"]
                  }
            ## Definning location of file
            import os
            if dir_file==[]:
                p = {"modfile"  : "cfg/" + s_name_model, #This the file of the model
                     "workdir"  : os.getcwd(),
                     "bifpar"   : bifpar
                     }
            else:
                p = {"modfile"  : "cfg/" + s_name_model, #This the file of the model
                     "workdir"  : os.getenv("HOME") + dir_file,
                     "bifpar"   : bifpar
                     }

            ##### getting expressions for currents, membrane potentials and odes
            sde, currents, mem_pot, s_description, d_inicond, d_pars, ss_currents = load_mod(p["modfile"],p['bifpar'].keys())
            ##### Defining units as 1 to replace later (erase them..)
            baseunits2 = [('mV', 1), ('ms', 1),('second', 1), ('cm2', 1), ('cm3', 1), ('uF', 1), ('psiemens', 1), ('um2', 1), ('msiemens', 1), ('cm', 1), ('kelvin', 1), ('mM', 1), ('mol', 1), ('uA', 1), ('mjoule', 1), ('coulomb',1), ('ufarad',1)]
            ##### Replacing units for ones.. in the expressions..
            varrhs = [(i,sympy.S(j).subs(baseunits2))
                            for i,j in sde]
            currwounits = [(i,sympy.S(j).subs(baseunits2))
                            for i,j in currents]
            sscurrwounits = [(i,sympy.S(j).subs(baseunits2))
                            for i,j in ss_currents]
            mempwounits = [(i,sympy.S(j).subs(baseunits2))
                            for i,j in mem_pot]

            inicondwounits = [(i,sympy.S(j).subs(baseunits2))
                            for i,j in d_inicond]
            #### Removing units from pars
            parslist = dict([(i,float(str(S(j).subs(baseunits2)))) for i,j in d_pars.items()])
            #### Separating definitions frpm expressions..
            if mempwounits==[]:
                s_mempot=[]
                mempotexp=[]
            if mempwounits!=[]:
                s_mempot,mempotexp = zip(*mempwounits)
            if currwounits==[]:
                s_curr=[]
                currexp=[]
            if currwounits!=[]:
                s_curr,currexp = zip(*currwounits)
            if sscurrwounits==[]:
                s_sscurr=[]
                sscurrexp=[]
            if sscurrwounits!=[]:
                s_sscurr,sscurrexp = zip(*sscurrwounits)

            s_svars,svarsexp = zip(*varrhs)

            self.s_pars=[j for j,k in bifpar.items()]
            self.svarsexp=svarsexp
            self.currexp=currexp
            self.sscurrexp=sscurrexp
            self.mempotexp=mempotexp
            self.baseunits=baseunits2
            self.s_model_tag=s_description
            self.s_model_reference_loc=p["workdir"]
            self.s_model_reference=p["modfile"]
            self.p = parslist
            self.n_state_vars=len(s_svars)
            self.s_state_vars=[j for j in s_svars]
            self.s_currents=[j for j in s_curr]
            self.s_sscurrents=[j for j in s_sscurr]
            self.s_concentrations=[]
            self.s_resting_membrane_potentials=[j for j in s_mempot]
            self.current_state=[float(dict(inicondwounits)[j]) for j in self.s_state_vars]
            self.ode_solver="odeint"
            self.noisy_current=[]
            for i_s in self.s_state_vars:
                if '_i' in i_s:
                    self.s_concentrations.append(i_s)
                if '_o' in i_s:
                    self.s_concentrations.append(i_s)

        def changing_pars(self,bifpar,pars4auto=False,strIapp='I_app'):
            ### Function that changes parameters (Only changes pars in the instance that calls it)
            ## bifpar= dictionary of parameters to be changed and their values
            if strIapp in bifpar:
                pass
            else:
                bifpar[strIapp]=[str(-5)+"* uA/cm2"]

            p = {"modfile"  : self.s_model_reference, #"cfg/wangBuzsaki_brian.json",      #This is our model
                 "workdir"  : self.s_model_reference_loc,
                 "bifpar"   : bifpar
                 }
            ##### Recalculating expressions for currents, membrane potentials and odes
            sde, currents, mem_pot,s_description, d_inicond, d_pars, ss_currents = load_mod(p["modfile"],p['bifpar'].keys())
            baseunits2=self.baseunits
            ### Recalculating expressions
            bifpar2=dict([(j,k[0]) for j,k in bifpar.items()])
            if pars4auto:
                for i_p in bifpar.keys():
                    bifpar2.pop(i_p, None)

            if strIapp in bifpar2:
                bifpar2.pop(strIapp, None)

            varrhs = [(i,sympy.S(str(sympy.S(j).subs(bifpar2))).subs(baseunits2))
                            for i,j in sde]
            currwounits = [(i,sympy.S(str(sympy.S(j).subs(bifpar2))).subs(baseunits2))
                            for i,j in currents]
            mempwounits = [(i,sympy.S(str(sympy.S(j).subs(bifpar2))).subs(baseunits2))
                            for i,j in mem_pot]
            sscurrwounits=[(i,sympy.S(str(sympy.S(j).subs(bifpar2))).subs(baseunits2))
                            for i,j in ss_currents]
            #### Separating definitions from expressions..
            if mempwounits==[]:
                s_mempot=[]
                mempotexp=[]
            if mempwounits!=[]:
                s_mempot,mempotexp = zip(*mempwounits)
            if currwounits==[]:
                s_curr=[]
                currexp=[]
            if currwounits!=[]:
                s_curr,currexp = zip(*currwounits)
            if sscurrwounits==[]:
                s_sscurr=[]
                sscurrexp=[]
            if sscurrwounits!=[]:
                s_sscurr,sscurrexp = zip(*sscurrwounits)

            s_svars,svarsexp = zip(*varrhs)

            #### adding I_app as par again.. so that instance it can be stimulated afterwards
            I_app=0
            bifpar_temp={}
            bifpar_temp= {
              strIapp : [str(I_app)+"* uA/cm2"]
              }
            #### Removing the annoying u before string
            s_pars=[j for j,k in bifpar_temp.items()]
            s_svars=[j for j in s_svars]
            s_curr=[j for j in s_curr]
            ## Adding the changed parameters to the parameters list again..
            for s_i,s_bi in bifpar2.items():
                d_pars[s_i]=s_bi
            ## Removing units again
            parslist = dict([(i,float(str(S(j).subs(baseunits2)))) for i,j in d_pars.items()])
            self.s_state_vars=s_svars
            self.s_pars=s_pars
            self.s_curr=s_curr
            self.mempotexp=mempotexp
            self.svarsexp=svarsexp
            self.currexp=currexp
            self.p = parslist

        def resting_membrane_potentials_fun(self):
            sym_svars=symbols(self.s_state_vars)
            fun_mempot=lambdify(sym_svars,self.mempotexp)
            return fun_mempot

        def neuron_currents_fun(self):
            sym_svars=symbols(self.s_state_vars)
            fun_currents = lambdify(sym_svars,self.currexp)
            return fun_currents

        def neuron_sscurrents_fun(self):
            sym_svars=symbols(self.s_state_vars)
            fun_sscurrents = lambdify(sym_svars,self.sscurrexp)
            return fun_sscurrents

        def neuron_changing_state_fun(self):
            # import pdb; pdb.set_trace()
            sym_spars=symbols(self.s_pars)
            sym_svars=symbols(self.s_state_vars)
            fun_ode =lambdify([i for i in sym_svars]+[i for i in sym_spars],self.svarsexp)
            return fun_ode

        def resting_membrane_potentials(self,v_State_Vars):
            fun_mempot=self.resting_membrane_potentials_fun()
            return fun_mempot(*v_State_Vars)

        def neuron_currents(self,v_State_Vars):
            fun_currents=self.neuron_currents_fun()
            return fun_currents(*v_State_Vars)

        def neuron_sscurrents(self,v):
            fun_sscurrents=self.neuron_sscurrents_fun()
            return fun_sscurrents(*[0,0,0,v])

        def neuron_changing_state(self,v_State_Vars,t,I_app,fun_ode):
            n_I=I_app(t)
            return fun_ode(* v_State_Vars.tolist()+[n_I])

        def stimulate_neuron(self,t,c_ini,I_stim):
            fun_ode=self.neuron_changing_state_fun()
            sol= odeint(self.neuron_changing_state, c_ini, t, args=(I_stim,fun_ode,))
            s_results=[]
            s_results=copy(self.s_state_vars)
            s_results.append("t")
            t_prov=t[...,None]
            v_results=np.append(sol,t_prov,1)
            return s_results, v_results

##### To call it neuron=generic_neuron_from_json('MTM_W_PNAs_Temp_snapshot_constleak'+'.json')
