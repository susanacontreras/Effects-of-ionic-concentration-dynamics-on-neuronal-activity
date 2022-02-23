from scipy.signal import butter, lfilter, freqz

### Class simulation used to store simulationss..
class Simulation():
        def __init__(self):
            self.c_neuron=[]#Link to neuron or actual instance (Should contain model and parameters used)
            self.s_Description=[]#Couple of lines describing the purpose of the simulations
            self.d_Protocol={
                        's_Type':[],
                        's_Stimulus_features':[],
                        'v_Stimulus_features':[]
                            }
            self.d_Configuration={
                        'n_max_step_integration':[],
                        'v_time_integration':[],
                        's_ODE_Solver':[],
                        'n_compressed_storage_factor':[]
                                }
            self.d_dinSys=[]
            self.d_dinSys={
                        'b_Convergence':[],
                        'n_tConvergence':[],
                        'Vars_ss':[],
                        'Stable_ss':[],
                        'Unstable_ss':[],
                        's_Classification':[],
                        'm_Jacobian4EqPoints':[],
                        's_Stability4EqPoints':[],
                        'm_Vect_Field':[],
                        'n_resol_Vect_Field':[],
                        'n_tempresol_Vect_Field':[],
                        'm_Vect_Field_aprox_trajectory':[],
                        'sv_state_vars_vectfield':[],
                        'm_Vect_Field_firing_mode':[],
                        'n_aprox_trajectory_converg_resol':[],
                        'm_fr_overview':[]
                            }
            self.a_Results=[]#Results of simulation
            self.m_Results_dinSys=[]#Results of simulation
            self.fr=[]#Results of simulation
            self.Adaptation=[]#If adaptation present.. Some parameters
            self.Adaptation={
                        'popt':[],
                        't_peak':[],
                        't_adapt':[],
                            }


class Results():
    def __init__(self,s_Results,v_Results,compress=[],fr=[]):#s_Results contains the strings naming results, and v_Results the actual reults to be stored
        a,b=shape(v_Results)
        c=len(s_Results)
        if compress==[]:
            count=0
            if b==c:
                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[:,count])
                    count+=1
            if a==c:
                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[count,:])
                    count+=1
            if a==b:
                print("Warning, in Results storage.. columns are taken as the different results")
                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[:,count])
                    count+=1
            if c!=a and c!=b:
                print("Sorry! couldn't create array of results because s_Results (String of features) doesn't have the same number of entries as v_Results (Matrix with the features)")
                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[:,count])
                    count+=1
        else:
            count=0
            i_spl=[]
            i_splv=[]
            if fr!=[]:
                i_sp=fr[2]
                if i_sp!=[]:
                    i_spl=i_sp.tolist()
                i_splv=i_spl
                for i in range(-compress,compress):
                    if i!=0 and i_sp!=[]:
                        i_sptemp=i_sp+i
                        i_spl=i_spl+i_sptemp.tolist()

            if b==c:
                v_normal=arange(0,a-1,compress).tolist()
                v_fin=v_normal+i_spl
                v_fin1=list(set(v_fin))
                v_fin2=sort(v_fin1)

                vect_i_sp=[]
                for i in i_splv:
                    vect_i_sp.append(v_fin2.tolist().index(i))

                fr1=[]
                fr1.append(fr[0])
                fr1.append(fr[1])
                fr1.append(vect_i_sp)
                setattr(self, 'fr', fr1)

                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[v_fin2,count])
                    count+=1
            if a==c:
                v_normal=arange(0,b-1,compress).tolist()
                v_fin=v_normal+i_spl
                v_fin1=list(set(v_fin))
                v_fin2=sort(v_fin1)

                vect_i_sp=[]
                for i in i_splv:
                    vect_i_sp.append(v_fin2.tolist().index(i))
                fr1=[]
                fr1.append(fr[0])
                fr1.append(fr[1])
                fr1.append(vect_i_sp)
                setattr(self, 'fr', fr1)

                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[v_fin2,count])
                    count+=1
            if a==b:
                print("Warning, in Results storage.. columns are taken as the different results")
                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[:,count])
                    count+=1
            if c!=a and c!=b:
                print("Sorry! couldn't create array of results because s_Results (String of features) doesn't have the same number of entries as v_Results (Matrix with the features)")
                while count<len(s_Results):
                    setattr(self, s_Results[count], v_Results[:,count])
                    count+=1

class Simulation_Light():
        def __init__(self):
            self.c_neuron=[]#Link to neuron or actual instance (Should contain model and parameters used)
            self.fr={
                        't':[],
                        'sp_c':[],
                        'sp_t':[],
                        't_I_stim':[],
                        's_I_stim':[],
                        'Fr':[]
                            }#Results of simulation
            self.state_vars={
                        'Na_i':[],
                        'K_i':[],
                        'K_o':[],
                        'i_p':[]
                            }#Results of simulation
            self.d_dinSys={
                        'saddle_node_Iapp':[],
                        'saddle_node_Iapp_error':[]
                            }#Results of simulation
            self.stim_ft={
                        's_Stimulus_features':[],
                        'v_Stimulus_features':[]
                            }
            self.d_Configuration={
                        'n_max_step_integration':[],
                        'v_time_integration':[],
                        's_ODE_Solver':[],
                        'n_compressed_storage_factor':[]
                                }

def static_input_simulation(neuron,exp_duration,step_size,interval0=0):
    ####Very fast define the stimulation protocol
    I_exp1 = lambda t: 0 if t<interval0 else step_size
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Constant input of magnitude "+str(step_size)+", to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Static_Input"
    sim.d_Protocol['s_Stimulus_features']=["Input_Magnitude","Exp_Duration"]
    sim.d_Protocol['v_Stimulus_features']=[step_size,exp_duration]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if t<"+str(interval0)+" else "+str(step_size)
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    ######And store them into the structure of results########
    sim.fr=[t_sp,fr,i_n]
    sim.a_Results=res
    return sim

def step_current_simulation(neuron,step_size,exp_duration,interval0=500,compress=[]):
    ####Very fast define the stimulation protocol
    I_exp1 = lambda t: 0 if t<interval0 else step_size
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Step current of magnitude "+str(step_size)+", to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Step_Current"
    sim.d_Protocol['s_Stimulus_features']=["Step_size","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[step_size,exp_duration,interval0]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if t<"+str(interval0)+" else "+str(step_size)
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
        sim.fr=res.fr
    ######And store them into the structure of results########
    sim.a_Results=res

    return sim

def step_current_simulation_small_interval(neuron,step_size,exp_duration,curr0=0,interval0=500,intervalf=3500,compress=[]):
    ####Very fast define the stimulation protocol
    I_exp1 = lambda t: curr0 if t<interval0 or t>intervalf else step_size
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Step current of magnitude "+str(step_size)+", to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Step_Current"
    sim.d_Protocol['s_Stimulus_features']=["Step_size","Exp_Duration","Time_Before_Step","Time_Step_Ends","Baseline_Current"]
    sim.d_Protocol['v_Stimulus_features']=[step_size,exp_duration,interval0,intervalf,curr0]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: "+str(curr0)+" if t<"+str(interval0)+" or t>"+str(intervalf)+" else "+str(step_size)
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
    ######And store them into the structure of results########
    sim.a_Results=res
    return sim

def step_current_simulation_fromNonZeroStart(neuron,ini_step_size,step_size,exp_duration,interval0=500):
    ####Very fast define the stimulation protocol
    I_exp1 = lambda t: ini_step_size if t<interval0 else step_size
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Step current of magnitude "+str(step_size)+", to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Step_Current"
    sim.d_Protocol['s_Stimulus_features']=["Step_size","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[step_size,exp_duration,interval0]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: "+str(ini_step_size)+" if t<"+str(interval0)+" else "+str(step_size)
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    ######And store them into the structure of results########
    sim.a_Results=res
    return sim

def ramp_current_simulation(neuron,step_size,exp_duration,interval0=500,interval_ramp=100,I0=[]):
    ####Very fast define the stimulation protocol
    if I0==[]:
        I_exp1 = lambda t: 0 if t<interval0 else step_size*t/interval_ramp
    else:
        I_exp1 = lambda t: I0 if t<interval0 else I0+(step_size-I0)*(t-interval0)/interval_ramp
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Step current of magnitude "+str(step_size)+", to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Step_Current"
    sim.d_Protocol['s_Stimulus_features']=["Step_size","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[step_size,exp_duration,interval0]
    if I0==[]:
        sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if t<"+str(interval0)+" else "+str(step_size)+"*t/"+str(interval_ramp)
    else:
        sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: "+str(I0)+" if t<"+str(interval0)+" else "+str(I0)+"+("+str(step_size)+"-"+str(I0)+")*(t-"+str(interval0)+")/"+str(interval_ramp)
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    ######And store them into the structure of results########
    sim.a_Results=res

    return sim

def ramp_current_simulation_discrete(neuron,step_size,exp_duration,interval0=500,interval_ramp=100,dt=20,I0=[]):
    ####Very fast define the stimulation protocol
    if I0==[]:
        I_exp1 = lambda t: 0 if t<interval0 else dt*int(t/dt)*step_size/interval_ramp
    else:
        I_exp1 = lambda t: I0 if t<interval0 else I0+dt*int((t-interval0)/dt)*(step_size-I0)/interval_ramp*dt
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Step current of magnitude "+str(step_size)+", to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Step_Current"
    sim.d_Protocol['s_Stimulus_features']=["Step_size","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[step_size,exp_duration,interval0]
    if I0==[]:
        sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if t<"+str(interval0)+" else "+str(dt)+"*int(t/"+str(dt)+")*"+str(step_size)+"/"+str(interval_ramp)
    else:
        sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: "+str(I0)+" if t<"+str(interval0)+" else "+str(I0)+"+"+str(dt)+"*int((t-"+str(interval0)+")/"+str(dt)+")*("+str(step_size)+"-"+str(I0)+")/"+str(interval_ramp)
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    ######And store them into the structure of results########
    sim.a_Results=res

    return sim


def sine_current_simulation(neuron,sine_base,sine_amp,sine_freq,exp_duration,interval0=500,compress=[]):
    ####Very fast define the stimulation protocol
    freq=sine_freq*0.001
    I_exp1 = lambda t: 0 if t<interval0 else sine_base+sine_amp/2+sine_amp/2*np.sin(2 * np.pi * freq * t + (np.pi/2))
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="Sine current of base "+str(sine_base)+", amplitude "+str(sine_amp)+" and frequency"+" to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Step_Current"
    sim.d_Protocol['s_Stimulus_features']=["sine_base","sine_amp","sine_freq","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[sine_base,sine_amp,sine_freq,exp_duration,interval0]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if t<"+str(interval0)+" else "+str(sine_base)+"+"+str(sine_amp/2)+"+"+str(sine_amp/2)+"*np.sin(2 * np.pi * "+str(freq)+"* t + (np.pi/2))"
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
        sim.fr=res.fr
    ######And store them into the structure of results########
    sim.a_Results=res
    return sim

def sine_current_simulation_small_interval(neuron,sine_base,sine_amp,sine_freq,exp_duration,interval0=500,intervalf=3500,compress=[]):
    ####Very fast define the stimulation protocol
    freq=sine_freq*0.001
    I_exp1 = lambda t: 0 if t<interval0 or t>intervalf else sine_base+sine_amp/2+sine_amp/2*np.sin(2 * np.pi * freq * t + (np.pi/2))
    t=np.linspace(0, exp_duration, exp_duration/resol)

    #####Initializa the structure to save results#########
    sim=Simulation()
    sim.c_neuron=neuron
    sim.s_Description="small interval stimulation with Sine current of base "+str(sine_base)+", amplitude "+str(sine_amp)+" and frequency"+" to "+neuron.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Sine_Current"
    sim.d_Protocol['s_Stimulus_features']=["sine_base","sine_amp","sine_freq","Exp_Duration","Time_Before_Step","intervalf"]
    sim.d_Protocol['v_Stimulus_features']=[sine_base,sine_amp,sine_freq,exp_duration,interval0,intervalf]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if t<"+str(interval0)+" or t>"+str(intervalf)+" else "+str(sine_base)+"+"+str(sine_amp/2)+"+"+str(sine_amp/2)+"*np.sin(2 * np.pi * "+str(freq)+"* t + (np.pi/2))"
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=neuron.ode_solver
    ######Run simulation################
    s_results, v_results = neuron.stimulate_neuron(t,neuron.current_state,I_exp1)
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass
    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
        sim.fr=res.fr
    ######And store them into the structure of results########
    sim.a_Results=res
    return sim

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


def noisy_current_simulation_var(neuron_in,noise_mean,noise_amp,noise_maxfreq,exp_duration,interval0=500,compress=[],light_version=0):
    ####Very fast define the stimulation protocol
    sim=Simulation()
    sim.c_neuron=copy(neuron_in)
    time_signal=exp_duration
    samples=int(np.ceil(time_signal*(1/resol)))
    noisy_signal_raw=band_limited_noise(0.0001, noise_maxfreq,samples,1.0/resol*1000)

    #noisy_signal=noise_mean+noise_amp/2.0/max(noisy_signal_raw)*noisy_signal_raw## This is wrong... the correct way is:
    noisy_signal=noise_mean+np.sqrt(noise_amp/np.var(noisy_signal_raw))*noisy_signal_raw## to get a signal with var=noise_amp
    sim.c_neuron.noisy_current=noisy_signal
    try:
        I_exp1 = lambda t: 0 if (t<=interval0 or t>time_signal) else sim.c_neuron.noisy_current[int((t-interval0)/resol)]
    except:
        print("Not working at t="+str(t)+" where int((t-interval0)/resol)="+str(int((t-interval0)/resol))+" for len(sim.c_neuron.noisy_current)="+str(len(sim.c_neuron.noisy_current)))
        raise
    #####Initializa the structure to save results#########

    sim.s_Description="Noisy current of mean "+str(noise_mean)+", amplitude "+str(noise_amp)+" and max frequency "+str(noise_maxfreq)+"Hz to "+neuron_in.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Noisy_Current"
    sim.d_Protocol['s_Stimulus_features']=["noise_mean","noise_amp","noise_maxfreq","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[noise_mean,noise_amp,noise_maxfreq,exp_duration,interval0]
    # I_exp1 = lambda t: 0 if t<interval0 else noisy_signal[int((t-interval0)/resol)]
    t=np.linspace(0, exp_duration, exp_duration/resol)
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if (t<="+str(interval0)+" or t>"+str(time_signal)+") else sim.c_neuron.noisy_current[int((t-"+str(interval0)+")/"+str(resol)+")]"
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=sim.c_neuron.ode_solver
    ######Run simulation################
    s_results, v_results = sim.c_neuron.stimulate_neuron(t,sim.c_neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass

    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
        sim.fr=res.fr
    ######And store them into the structure of results########
    if light_version==0:
        sim.a_Results=res
    if light_version==1:
        sim.a_Results=[]
    return sim

def noisy_current_simulation_var_plusDepPulses(neuron_in,noise_mean,noise_amp,noise_maxfreq,exp_duration,t_pulse=[700,1000],pulse_length=20.0,pulse_ampl=2.0,interval0=500,compress=[],light_version=0):
    ####Very fast define the stimulation protocol
    sim=Simulation()
    sim.c_neuron=copy(neuron_in)
    time_signal=exp_duration
    samples=int(np.ceil(time_signal*(1/resol)))
    noisy_signal_raw=band_limited_noise(0.0001, noise_maxfreq,samples,1.0/resol*1000)

    noisy_signal=noise_mean+noise_amp/2.0/max(noisy_signal_raw)*noisy_signal_raw
    for tt_pulse in t_pulse:
        for i_tlp in range(0,int(pulse_length/resol)):
            noisy_signal[int(tt_pulse/resol+i_tlp)]=noisy_signal[int(tt_pulse/resol+i_tlp)]+pulse_ampl

    sim.c_neuron.noisy_current=noisy_signal
    try:
        I_exp1 = lambda t: 0 if (t<=interval0 or t>time_signal) else sim.c_neuron.noisy_current[int((t-interval0)/resol)]
    except:
        print("Not working at t="+str(t)+" where int((t-interval0)/resol)="+str(int((t-interval0)/resol))+" for len(sim.c_neuron.noisy_current)="+str(len(sim.c_neuron.noisy_current)))
        raise
    #####Initializa the structure to save results#########

    sim.s_Description="Noisy current of mean "+str(noise_mean)+", amplitude "+str(noise_amp)+" and max frequency "+str(noise_maxfreq)+"Hz to "+neuron_in.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Noisy_Current"
    sim.d_Protocol['s_Stimulus_features']=["noise_mean","noise_amp","noise_maxfreq","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[noise_mean,noise_amp,noise_maxfreq,exp_duration,interval0]
    # I_exp1 = lambda t: 0 if t<interval0 else noisy_signal[int((t-interval0)/resol)]
    t=np.linspace(0, exp_duration, exp_duration/resol)
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if (t<="+str(interval0)+" or t>"+str(time_signal)+") else sim.c_neuron.noisy_current[int((t-"+str(interval0)+")/"+str(resol)+")]"
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=sim.c_neuron.ode_solver
    ######Run simulation################
    s_results, v_results = sim.c_neuron.stimulate_neuron(t,sim.c_neuron.current_state,I_exp1)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass

    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
        sim.fr=res.fr
    ######And store them into the structure of results########
    if light_version==0:
        sim.a_Results=res
    if light_version==1:
        sim.a_Results=[]
    return sim

def noisy_current_simulation_var_fKo(neuron_in,noise_mean,noise_amp,noise_maxfreq,exp_duration,interval0=500,Ko_freq=1.0,Ko_amp=3.0,Ko_mean=3.0,compress=[],light_version=0):
    ####Very fast define the stimulation protocol
    sim=Simulation()
    sim.c_neuron=copy(neuron_in)
    time_signal=exp_duration
    samples=int(np.ceil(time_signal*(1/resol)))
    noisy_signal_raw=band_limited_noise(0.0001, noise_maxfreq,samples,1.0/resol*1000)
    noisy_signal=noise_mean+noise_amp/2.0/max(noisy_signal_raw)*noisy_signal_raw
    sim.c_neuron.noisy_current=noisy_signal
    try:
        I_exp1 = lambda t: 0 if (t<=interval0 or t>time_signal) else sim.c_neuron.noisy_current[int((t-interval0)/resol)]
    except:
        print("Not working at t="+str(t)+" where int((t-interval0)/resol)="+str(int((t-interval0)/resol))+" for len(sim.c_neuron.noisy_current)="+str(len(sim.c_neuron.noisy_current)))
        raise
    ## Slow Ko wave
    try:
        f_Ko = lambda t: Ko_mean if (t<=interval0 or t>time_signal) else Ko_mean+Ko_amp*np.sin(2*np.pi*Ko_freq*(t-interval0)/1000.0-np.pi/2)
    except:
        print("Not working at t="+str(t)+" where int((t-interval0)/resol)="+str(int((t-interval0)/resol))+" for len(sim.c_neuron.noisy_current)="+str(len(sim.c_neuron.noisy_current)))
        raise
    #####Initializa the structure to save results#########
    sim.s_Description="Noisy current of mean "+str(noise_mean)+", amplitude "+str(noise_amp)+" and max frequency "+str(noise_maxfreq)+"Hz to "+neuron_in.s_model_tag+" parameters can be found with sim.self.c_neuron.p"
    sim.d_Protocol['s_Type']="Noisy_Current"
    sim.d_Protocol['s_Stimulus_features']=["noise_mean","noise_amp","noise_maxfreq","Exp_Duration","Time_Before_Step"]
    sim.d_Protocol['v_Stimulus_features']=[noise_mean,noise_amp,noise_maxfreq,exp_duration,interval0]
    # I_exp1 = lambda t: 0 if t<interval0 else noisy_signal[int((t-interval0)/resol)]
    t=np.linspace(0, exp_duration, exp_duration/resol)
    sim.c_neuron.Ko_wave=[f_Ko(ti) for ti in t]
    sim.d_Protocol['s_Executable_stimulus']="I_exp1 = lambda t: 0 if (t<="+str(interval0)+" or t>"+str(time_signal)+") else sim.c_neuron.noisy_current[int((t-"+str(interval0)+")/"+str(resol)+")]"
    sim.d_Configuration['n_max_step_integration']=resol
    sim.d_Configuration['v_time_integration']=[t[0],t[-1]]
    sim.d_Configuration['s_ODE_Solver']=sim.c_neuron.ode_solver
    ######Run simulation################
    s_results, v_results = sim.c_neuron.stimulate_neuron(t,sim.c_neuron.current_state,I_exp1,i_p_f=[],K_o_f=f_Ko)
    ######Organize Simulation results into a structure####
    res=Results(s_results, v_results)
    try:
        t_sp,fr,i_n=firing_rate(res.t,res.V)
    except:
        try:
            t_sp,fr,i_n=firing_rate(res.t,res.v)
        except:
            pass

    sim.fr=[t_sp,fr,i_n]
    if compress !=[]:
        res=Results(s_results, v_results,compress,sim.fr)
        sim.d_Configuration['n_compressed_storage_factor']=compress
        sim.fr=res.fr
    ######And store them into the structure of results########
    if light_version==0:
        sim.a_Results=res
    if light_version==1:
        sim.a_Results=[]
    return sim
