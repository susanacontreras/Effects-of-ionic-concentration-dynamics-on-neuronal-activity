import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

################important define!
global precision_convergence_ss

precision_convergence_ss=0.00001

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print( "Toc: start time not set")

def detect_peaks(signal, threshold=0.5, spacing=3):#https://github.com/MonsieurV/py-findpeaks/blob/master/tests/libs/tony_beltramelli_detect_peaks.py
    limit=None
    data=signal
    len = data.size
    x = np.zeros(len+2*spacing)
    x[:spacing] = data[0]-1.e-6
    x[-spacing:] = data[-1]-1.e-6
    x[spacing:spacing+len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start : start + len]  # before
        start = spacing
        h_c = x[start : start + len]  # central
        start = spacing + s + 1
        h_a = x[start : start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    peak_indexes = np.argwhere(peak_candidate)
    peak_indexes = peak_indexes.reshape(peak_indexes.size)
    if limit is not None:
        peak_indexes = ind[data[ind] > limit]
    return peak_indexes

def firing_rate(t,V):
    fr=np.zeros(size(t))
    t_sp=np.zeros(size(t))
    d=[]
    sp=[]
    i_n=[]
    d=detect_peaks(V)
    if d==[] or len(d)<2:
        fr=np.zeros(size(t))
        t_sp=np.zeros(size(t))
    else:
        sp=V[d]>thresh
        if np.count_nonzero(sp)>2:
            i=np.transpose(sp)*d
            i_n=i[np.nonzero(i)]
            t_sp=t[i_n]
            t_sp=t_sp.reshape(size(t_sp),1)
            fr=1/(np.diff(t_sp, n=1, axis=0)/1000)
            t_sp=t_sp[1:]
        else:
            fr=np.zeros(size(t))
            t_sp=t

    return t_sp,fr,i_n

def extract_spike_ampl(V, i_nn,compress=[]):
    c=0
    sp_V_max=[]
    sp_V_min=[]
    sp_ampl=[]
    if compress !=[]:
        i_n=[int(i/compress) for i in i_nn]
    else:
        i_n=i_nn
    d_i_ni=int(i_n[0]/2)

    for i in i_n:
        if c<len(i_n)-1:
            d_i_nd=int((i_n[c+1]-i)/2)
        else:
            d_i_nd=int((len(V)-i)/2)
        sp_V_max.append(max(V[i-d_i_ni:i+d_i_nd]))
        sp_V_min.append(min(V[i-d_i_ni:i+d_i_nd]))
        sp_ampl.append(max(V[i-d_i_ni:i+d_i_nd])-min(V[i-d_i_ni:i+d_i_nd]))
        c+=1
        d_i_ni=int((i_n[c-1]-i)/2)

    return sp_ampl,sp_V_max,sp_V_min

def create_vector_field(sv_state_vars,sv_condini_explore,m_discrete_pars_explore,n_neurons,s_model,d_pars,run_parallel=True,max_num_processes=20,max_num_local_processes=10,dir_data='/scratch/',file_n=[]):
#####This function runs several simulations of length time_stim (global) initialized at different initial conditions to create a vector field..
    local_hostname 		= 'compute1'	# name of host on which the limit is...

    def creating_vector(sim,sv_state_vars,sv_condini_explore):
        mat_ret1=[]
        mat_ret2=[]
        if isinstance(sim,Simulation):
            lvdC=len(getattr(sim.a_Results, sv_state_vars[0]))
            for i in sv_condini_explore:
                mat_ret1.append(sim.c_neuron.p[i])
            for i in sv_state_vars:
                v=getattr(sim.a_Results, i)
                mat_ret2.append(mean(v[-int(lvdC*0.1):-1])-mean(v[int(lvdC*0.1):int(lvdC*0.2)]))
            return mat_ret1+mat_ret2
        else:
            return []

    sv_Pars_explor=sv_condini_explore
    neurons_cini=create_neurons_discreteSampling(s_model,d_pars,sv_Pars_explor,n_neurons,m_discrete_pars_explore)
    sim_cini=run_parameter_exploration(neurons_cini,run_parallel,max_num_processes,local_hostname,max_num_local_processes)

    whole_mat_r=[]
    firing_mode=[]
    fr_overview_mean=[]
    fr_overview_change=[]
    c=0
    sim2save=[]
    for i in sim_cini:
        if isinstance(i,Simulation):
            mat_ret=creating_vector(i,sv_state_vars,sv_condini_explore)
            whole_mat_r.append(mat_ret)
            firing_mode.append(f_firing_mode(i))
            mean_fr, ini_fr, fin_fr=f_firing_vect(i)
            if size(mean_fr)>0:
                fr_overview_mean.append(mean_fr)
                fr_overview_change.append(fin_fr-ini_fr)
            else:
                fr_overview_mean.append(0)
                fr_overview_change.append(0)
            sim2save.append(i)
        c+=1
    whole_mat=np.matrix(whole_mat_r)
    if file_n!=[]:
        save_experiment(sim2save,dir_data,file_n)

    return whole_mat, firing_mode, fr_overview_mean, fr_overview_change


def f_firing_mode_old(sim):
    #### Determines firing mode of neuron.. 1: silent, 2: firing, 3: DepBlock, if firing rate <1, the function looks at V, if it is <50 is at rest, otherwise dep-Block
    if sim.fr != []:
        if sum(sim.fr[1])<=0:
            if sim.a_Results.V[-1]<-50:
                mode=1
            else:
                mode=3
        if sum(sim.fr[1])>0 and sim.fr[0][-1]>=sim.a_Results.t[-int(len(sim.a_Results.t)*0.1)]:
            if sim.fr[1][-1]>1:
                mode=2
            else:
                if sim.a_Results.V[-1]<-50:
                    mode=1
                else:
                    mode=3
        if sum(sim.fr[1])>0 and sim.fr[0][-1]<sim.a_Results.t[-int(len(sim.a_Results.t)*0.1)]:
            mode=3
    else:
        if sim.a_Results.V[-1]<-50:
            mode=1
        else:
            mode=3
    return mode

def f_firing_mode(sim):
    #### Determines firing mode of neuron.. 1: silent, 2: firing, 3: DepBlock, the function looks at V, if it is <-40 is at rest, otherwise dep-Block
    if sim.fr != []:
        if sum(sim.fr[1])<=0:
            if sim.a_Results.V[-1]<-40:
                mode=1
            else:
                mode=3
        if sum(sim.fr[1])>0:
            tt_c=[]
            tt_c=np.nonzero((sim.a_Results.t>sim.fr[0][-1]))[0]
            if tt_c!=[]:
                if 1000/(sim.a_Results.t[-1]-sim.fr[0][-1])<sim.fr[1][-1]:
                    if mean(sim.a_Results.V[tt_c])<-40:
                        mode=2
                    else:
                        mode=3
                else:
                    mode=2
            else:
                if sim.a_Results.t[-1]==sim.fr[0][-1] and len(sim.fr[0])>2:
                    if 1000/(sim.a_Results.t[-1]-sim.fr[0][-2])<sim.fr[1][-2]:
                        if sim.a_Results.V[-1]<-40:
                            mode=2
                        else:
                            mode=3
                    else:
                        mode=2
                else:
                    if sim.a_Results.V[-1]<-40:
                        mode=1
                    else:
                        mode=3
    else:
        if sim.a_Results.V[-1]<-40:
            mode=1
        else:
            mode=3
    return mode


def f_firing_vect(sim):
    mean_fr=[]
    ini_fr=[]
    fin_fr=[]
    if sim.fr != []:
        if sum(sim.fr[1])> 0:
            mean_fr=mean(sim.fr[1])
            ini_fr1=sim.fr[1][0]
            ini_fr=ini_fr1[0]
            fin_fr1=sim.fr[1][-1]
            fin_fr=fin_fr1[0]
    return mean_fr, ini_fr, fin_fr

def f_phase_plane_descpt(INFO,i_ap):
    ###### descpt=0 Uni-stable-silent: means that there is one one stable fixed point on the phase phase Plane
    ###### descpt=1 Uni-stable-spiking: means that there is one one stable orbit on phase Plane
    ###### descpt=2 Bi-stable: means that there is a stable fixed point and a stable orbit on phase Plane
    descpt=[]
    s_orbit=[]
    s_fp=[]
    oc_pars=[]
    iiv=[]

    try:
        for i_s,j_s in INFO.items():
            if 'Jac_info' in i_s:
                iiv.append(int(i_s[i_s.index('o')+1:]))

        for ii in iiv:
            # import pdb; pdb.set_trace()
            one_lc_iapp=[]
            if abs(i_ap-INFO['node_pars'+str(ii)]['I_app'])<0.1:
                pars=INFO['node_pars'+str(ii)]
                saddle_point=INFO['Jac_info'+str(ii)][0]
                m_Jmat=INFO['Jac_info'+str(ii)][1]
                Jeival=INFO['Jac_info'+str(ii)][2]
                Jeivect=INFO['Jac_info'+str(ii)][3]
                ### Looking for a stable orbit
                try:
                    c_lc_iap=0
                    for ilc_iap in INFO['One_lc_iapp']:
                        if abs(i_ap-ilc_iap)<0.1:
                            one_lc_iapp=c_lc_iap
                            oc_pars=copy(pars)
                        c_lc_iap+=1
                except:
                    pass
                ### Looking for a stable fixed point
                if any(Jeival.real>0):
                    pass
                else:
                    s_fp=1
                    oc_pars=copy(pars)

                if one_lc_iapp==[]:
                    s_orbit=[]
                else:
                    s_orbit=1

        if s_orbit==[] and s_fp==1:
            descpt=0
        if s_orbit==1 and s_fp==[]:
            descpt=1
        if s_orbit==1 and s_fp==1:
            descpt=2
        if s_orbit==[] and s_fp==[]:
            descpt=[]
    except:
        pass

    return descpt,oc_pars

def f_HB_Iapp_lcOnset_Iapp(INFO):
    ###### HB_Iapp: Current to reach dep block
    ###### lc_Iapp: Current to start spiking -Spiking threshold
    HB_Iapp=[]
    lc_Iapp=[]
    try:
        HB_Iapp=INFO['HBpoint']['I_app']
    except:
        pass
    try:
        lc_Iapp=INFO['lcpoint']['I_app']
    except:
        pass
    return HB_Iapp, lc_Iapp
