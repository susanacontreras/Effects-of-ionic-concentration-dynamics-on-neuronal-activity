#################Example for creating subplots
    # f1, ax= plt.subplots(2, sharex=True, facecolor="1",figsize=(10,12))
    # rc('font', family='serif', size=18)
    # ax[0]=plot_t_vs_V(sim.a_Results,ax[0])
    # ax[1]=plot_t_vs_Concentrations(sim.a_Results,ax[1])
    # show()
#################Example to create single plot
# f1, ax= plt.subplots(1, sharex=True, facecolor="1",figsize=(10,12))
# rc('font', family='serif', size=18)
# ax=plot_t_vs_V(sim.a_Results,ax)
# show()
import matplotlib
from matplotlib import rcParams
matplotlib.rcParams['text.usetex'] = True
# # matplotlib.rcParams['text.latex.unicode'] = True
# rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import pylab as pl
from copy import copy


global size_title_font, size_axis_font, size_axisnumbers_font, size_legend_font, fig_wide, fig_height
size_title_font=50
size_axis_font=40
size_axisnumbers_font=30
size_legend_font=30
fig_wide=15
fig_height=15
matplotlib.rc('font', family='sans-serif', size=size_axis_font)

def plot_t_vs_V(sim,axarr,s_Optional_title="",c=[0,0,0],reset_t=[],ms2s=[]):
###### Axes Voltage trace#########################################
    for i in sim.c_neuron.s_state_vars:
        if i=='V':
            V=sim.a_Results.V
        if i=='v':
            V=sim.a_Results.v
    t=sim.a_Results.t
    if reset_t!=[]:
        if ms2s!=[]:
            axarr.plot((t-reset_t)/1000., V,color=c)
        else:
            axarr.plot(t-reset_t, V,color=c)
    else:
        if ms2s!=[]:
            axarr.plot(t/1000., V,color=c)
        else:
            axarr.plot(t, V,color=c)
    axarr.set_title(r'Voltage trace '+s_Optional_title,fontsize=size_title_font)
    axarr.set_ylabel(r'V [mV]',fontsize=size_axis_font)
    # axarr.set_xlabel(r'Time [ms]',fontsize=size_axis_font)
    # axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_t_vs_Concentrations(sim,axarr,s_Optional_title="",reset_t=[]):
###### Axes Concentrations#########################################
    if reset_t!=[]:
        rt=0
    else:
        rt=reset_t
    axarr.set_title(r'Ionic Concentrations '+s_Optional_title,fontsize=size_title_font)
    for i in sim.c_neuron.s_concentrations:
        axarr.plot(sim.a_Results.t-rt, getattr(sim.a_Results,i),linewidth=3,label='$'+i+'$')
    axarr.set_ylabel(r'[mM]',fontsize=size_axis_font)
    axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.legend(loc=1,prop={'size': size_legend_font})
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_t_vs_specf_Concentrations(sim,s_concentrations,axarr,s_Optional_title=""):
###### Axes Concentrations#########################################
    axarr.set_title(r'Ionic Concentrations '+s_Optional_title,fontsize=size_title_font)
    for i in s_concentrations:
        c=0
        for j in sim.c_neuron.s_concentrations:
            if i==j:
                axarr.plot(sim.a_Results.t, getattr(sim.a_Results,i),linewidth=5,label='$'+i+'$')
            c+=1

    axarr.set_ylabel(r'[mM]',fontsize=size_axis_font)
    # axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.legend(loc=1,prop={'size': size_legend_font})
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr


def plot_t_vs_Currents(sim,axarr,s_Optional_title=""):
###### Axes Concentrations#########################################
    axarr.set_title(r'Currents'+s_Optional_title,fontsize=size_title_font)
    I_vv=[]
    c=0
    for i in sim.a_Results.t:
        inst_state= [None] * len(sim.c_neuron.s_state_vars)
        cc=0
        for j in sim.c_neuron.s_state_vars:
            inst_state[cc]=(getattr(sim.a_Results,j)[c])
            cc+=1
        I_vv.append(sim.c_neuron.neuron_currents(inst_state))
        c+=1
    I_vm=matrix(I_vv)

    c=0
    for j in sim.c_neuron.s_currents:
        axarr.plot(sim.a_Results.t, I_vm[:,c],linewidth=3,label='$'+j+'$')
        c+=1
    axarr.set_ylabel(r'I[uA]',fontsize=size_axis_font)
    axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.legend(loc=1,prop={'size': size_legend_font})
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr


def plot_t_vs_specf_Currents(sim,s_currents,axarr,s_Optional_title=""):
    axarr.set_title(r'Currents'+s_Optional_title,fontsize=size_title_font)
    I_vv=[]
    c=0
    for i in sim.a_Results.t:
        inst_state= [None] * len(sim.c_neuron.s_state_vars)
        cc=0
        for j in sim.c_neuron.s_state_vars:
            inst_state[cc]=(getattr(sim.a_Results,j)[c])
            cc+=1
        I_vv.append(sim.c_neuron.neuron_currents(inst_state))
        c+=1
    I_vm=matrix(I_vv)

    for i in s_currents:
        c=0
        for j in sim.c_neuron.s_currents:
            if i==j:
                axarr.plot(sim.a_Results.t, I_vm[:,c],linewidth=3,label='$'+i+'$')
            c+=1
    axarr.set_ylabel(r'I[uA]',fontsize=size_axis_font)
    axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.legend(loc=1,prop={'size': size_legend_font})
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_t_vs_RestingPotentials(sim,axarr,s_Optional_title="",reset_t=[],ms2s=[]):
###### Axes Concentrations#########################################
    for i in sim.c_neuron.s_state_vars:
        if i=='V':
            V=sim.a_Results.V
        if i=='v':
            V=sim.a_Results.v

    E_vv=[]
    c=0
    for i in sim.a_Results.t:
        inst_state= [None] * len(sim.c_neuron.s_state_vars)
        cc=0
        for j in sim.c_neuron.s_state_vars:
            inst_state[cc]=(getattr(sim.a_Results,j)[c])
            cc+=1
        E_vv.append(sim.c_neuron.resting_membrane_potentials(inst_state))
        c+=1
    E_vm=matrix(E_vv)

    c=0
    for j in sim.c_neuron.s_resting_membrane_potentials:
        axarr.plot(sim.a_Results.t, E_vm[:,c],linewidth=3,label='$'+j+'$')
        c+=1
    axarr.plot(sim.a_Results.t,V,color=[0,0,0],label="V")
    axarr.set_title(r''+s_Optional_title,fontsize=size_title_font)
    axarr.set_ylabel(r'V[mV]',fontsize=size_axis_font)
    # axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.legend(loc=1,prop={'size': size_legend_font})
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_t_vs_specfRestingPotentials(sim,s_membpot,axarr,s_Optional_title="",reset_t=[],ms2s=[]):
###### Axes Concentrations#########################################
    for i in sim.c_neuron.s_state_vars:
        if i=='V':
            V=sim.a_Results.V
        if i=='v':
            V=sim.a_Results.v

    E_vv=[]
    c=0
    for i in sim.a_Results.t:
        inst_state= [None] * len(sim.c_neuron.s_state_vars)
        cc=0
        for j in sim.c_neuron.s_state_vars:
            inst_state[cc]=(getattr(sim.a_Results,j)[c])
            cc+=1
        E_vv.append(sim.c_neuron.resting_membrane_potentials(inst_state))
        c+=1
    E_vm=matrix(E_vv)

    c=0
    for j in sim.c_neuron.s_resting_membrane_potentials:
        if j in s_membpot:
            axarr.plot(sim.a_Results.t, E_vm[:,c],linewidth=5,label='$'+j+'$')
        c+=1
    axarr.plot(sim.a_Results.t,V,color=[0,0,0],label="V")
    axarr.set_title(r''+s_Optional_title,fontsize=size_title_font)
    axarr.set_ylabel(r'V[mV]',fontsize=size_axis_font)
    # axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.legend(loc=1,prop={'size': size_legend_font})
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_t_vs_Fr(t,fr,axarr,s_Optional_title="",c=[128/255.0,128/255.0,128/255.0]):
###### Axes Voltage trace#########################################
    axarr.set_title(r'Firing Rate '+s_Optional_title,fontsize=size_title_font)
    if len(fr)>0:
        axarr.plot(t, fr,color=c,linewidth=4)
    else:
        axarr.plot(t, np.zeros(len(t)),color=c,linewidth=4)
    axarr.set_ylabel(r'Fr [Hz]',fontsize=size_axis_font)
    # axarr.set_xlabel( r'Time [ms]',fontsize=size_axis_font)
    # axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_sp_timing(t_sp,axarr,s_Optional_title="",c=[128/255.0,128/255.0,128/255.0]):
    axarr.set_title(r'Spike Occurence'+s_Optional_title,fontsize=size_title_font)
    axarr.vlines(t_sp,0,1)
    axarr.get_yaxis().set_visible(False)

def plot_t_vs_Iapp(t,I,axarr,s_Optional_title="",c=[128/255.0,128/255.0,128/255.0],no_title=False):
###### Axes Voltage trace#########################################
    axarr.set_title(r'Input current '+s_Optional_title,fontsize=size_title_font)
    if no_title:
        axarr.set_title(r''+s_Optional_title,fontsize=size_title_font)
    axarr.plot(t, I,color=c,linewidth=4)
    axarr.set_ylabel(r'I [uA]',fontsize=size_axis_font)
    axarr.set_xlabel( r'Time [ms]',fontsize=size_axis_font)
    # axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr

def plot_fi_curve(f,I,axarr,s_Optional_title="",c=[90/255.0,90/255.0,90/255.0]):
    axarr.set_title(r'f-I curve'+s_Optional_title,fontsize=size_title_font)
    axarr.plot(I,f,color=c)
    axarr.set_ylabel(r'F [Hz]',fontsize=size_axis_font)
    axarr.set_xlabel(r'I [uA]',fontsize=size_axis_font)
    axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.tick_params(labelsize=size_axisnumbers_font)
    return axarr


def plot_oneSim_specf_currents(sim,s_currents,s_Optional_title="",t_i=[],I_i=[],reset_t=[],w_Iapp=[]):
    #####Inputs: sim: Simuation instance, containning a simulation data
    #            s_Optional_title: Optional title of the figure
    #            t_i: optional Time span vector of the observation t_i=[t0,tf]
    f1, ax= plt.subplots(3, sharex=True, facecolor="1",figsize=(fig_wide,fig_height))
    if reset_t!=[]:
        rt=reset_t
    else:
        rt=0
    ax[0]=plot_t_vs_V(sim,ax[0])
    ax[0].set_title(''+s_Optional_title,fontsize=size_title_font)
    plt.grid(b=False)
    if sim.fr!=[]:
        ax[1]=plot_t_vs_Fr(sim.fr[0],sim.fr[1],ax[1])
    else:
        ax[1]=plot_t_vs_Fr(sim.a_Results.t,[],ax[1])
    plt.grid(b=False)
    if w_Iapp!=[]:
        exec(sim.d_Protocol['s_Executable_stimulus'])
        I=[I_exp1(i)  for i in sim.a_Results.t]
        ax[2]=plot_t_vs_Iapp(sim.a_Results.t,I,ax[2])
    if I_i!=[]:
        ax[2].set_ylim(I_i)
    plt.grid(b=False)
    ax[2]=plot_t_vs_specf_Currents(sim,s_currents,ax[2])
    ax[0].tick_params(labelsize=size_axisnumbers_font)
    ax[1].tick_params(labelsize=size_axisnumbers_font)
    ax[2].tick_params(labelsize=size_axisnumbers_font)
    if t_i==[]:
        plt.autoscale(enable=True, axis='x', tight=True)
    else:
        plt.xlim(t_i)

    plt.grid(b=False)
    return f1

def plot_oneSim_WConc_nice(sim,lt=[]):
    if lt==[]:
        lt=len(sim.a_Results.t)-1
    f1 = plt.figure(facecolor="1",figsize=(fig_wide,fig_height))
    ax=[]
    ax.append(plt.subplot2grid((15, 1), (0, 0), rowspan=7))
    ax.append(plt.subplot2grid((15, 1), (7, 0), sharex=ax[0]))
    ax.append(plt.subplot2grid((15, 1), (8, 0), rowspan=7, sharex=ax[0]))
    ##### Plot voltage trace
    ax[0].set_title(r'Voltage trace ',fontsize=size_title_font,y=1.08)
    ax[0].set_ylabel(r'V [mV]',fontsize=size_axis_font,labelpad=10)
    ax[0].plot(sim.a_Results.t[0:lt],sim.a_Results.V[0:lt],color=c)
    ax[0].get_xaxis().set_visible(False)
    locatory = MaxNLocator(nbins=3) # with 3 bins you will have 4 ticks
    ax[0].yaxis.set_major_locator(locatory)
    ax[0].tick_params(axis='y', pad=10)
    plt.locator_params(nbins=4)

    ##### Plot Stimulus
    exec(sim.d_Protocol['s_Executable_stimulus'])
    I=[I_exp1(i)  for i in sim.a_Results.t]
    ax[1].plot(sim.a_Results.t[0:lt], I[0:lt],color='silver',linewidth=1)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax2 = ax[1].twinx()
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylabel("Input",color="silver",labelpad=10)

    # ax[1].set_ylabel(r'I [uA]',fontsize=size_axis_font,labelpad=10)
    locatory1 = MaxNLocator(nbins=2) # with 3 bins you will have 4 ticks
    ax[1].yaxis.set_major_locator(locatory1)
    ax[1].spines['bottom'].set_color('silver')
    ax[1].spines['top'].set_color('silver')
    ax[1].spines['left'].set_color('silver')
    ax[1].spines['right'].set_color('silver')
    for t in ax[1].xaxis.get_ticklines(): t.set_color('silver')
    ax2.yaxis.set_major_locator(locatory1)
    ax2.spines['bottom'].set_color('silver')
    ax2.spines['top'].set_color('silver')
    ax2.spines['left'].set_color('silver')
    ax2.spines['right'].set_color('silver')

    ##### Plot ionic concentrations
    ax[2].set_title(r'Ionic Concentrations ',fontsize=size_title_font,y=1.08)
    v_concs=['K_o','Na_i']
    for i in v_concs:
    	if i=='K_o':
            	ax[2].plot(sim.a_Results.t[0:lt], getattr(sim.a_Results,i)[0:lt],linewidth=4,color='blue',label='$'+i+'$')
    	if i=='Na_i':
    		ax[2].plot(sim.a_Results.t[0:lt], getattr(sim.a_Results,i)
    [0:lt],linewidth=4,color='red',label='$'+i+'$')
    ###### Make fonts bigger,
    plt.locator_params(nbins=4)
    ax[2].set_ylabel(r'C [mM]',fontsize=size_axis_font,labelpad=40)
    # ax[1].set_xlabel(r't[ms]',fontsize=size_axis_font,labelpad=20)
    ax[2].set_xlabel(r't [ms]',fontsize=size_axis_font,labelpad=10)
    ax[2].tick_params(axis='x', pad=10)
    ax[2].tick_params(axis='y', pad=10)
    ###### and adding legend
    locatory2 = MaxNLocator(nbins=3) # with 3 bins you will have 4 ticks
    ax[2].yaxis.set_major_locator(locatory2)
    plt.tight_layout()
    box0 = ax[0].get_position()
    box1 = ax[1].get_position()
    box2 = ax[2].get_position()
    ax[0].set_position([box0.x0, box0.y0, box0.width * 0.8, box0.height])
    ax[1].set_position([box1.x0, box0.y0*0.99, box1.width * 0.8, box1.height])
    ax2.set_position([box1.x0, box0.y0*0.99, box1.width * 0.8, box1.height])
    ax[2].set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

    # Put a legend to the right of the current axis
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': size_legend_font})

    return f1
def plot_oneSim_WOConc(sim,s_Optional_title="",t_i=[],I_i=[],reset_t=[],ms2s=[]):
    #####Inputs: sim: Simuation instance, containning a simulation data
    #            s_Optional_title: Optional title of the figure
    #            t_i: optional Time span vector of the observation t_i=[t0,tf]
    f1, ax= plt.subplots(3, sharex=True, facecolor="1",figsize=(fig_wide,fig_height))
    if reset_t!=[]:
        rt=reset_t
    else:
        rt=0
    ax[0]=plot_t_vs_V(sim,ax[0],reset_t=reset_t,ms2s=ms2s)
    ax[0].set_title(''+s_Optional_title,fontsize=size_title_font)
    plt.grid(b=False)
    if sim.fr!=[]:
        if ms2s!=[]:
            ax[1]=plot_t_vs_Fr((sim.fr[0]-rt)/1000.,sim.fr[1],ax[1])
        else:
            ax[1]=plot_t_vs_Fr(sim.fr[0]-rt,sim.fr[1],ax[1])
    else:
        if ms2s!=[]:
            ax[1]=plot_t_vs_Fr((sim.a_Results.t-rt)/1000.,sim.fr[1],ax[1])
        else:
            ax[1]=plot_t_vs_Fr(sim.a_Results.t-rt,[],ax[1])
    plt.grid(b=False)
    sim=copy(sim)
    exec(sim.d_Protocol['s_Executable_stimulus'])
    I=[I_exp1(i)  for i in sim.a_Results.t]
    if ms2s!=[]:
        ax[2]=plot_t_vs_Iapp((sim.a_Results.t-rt)/1000.,I,ax[2])
    else:
        ax[2]=plot_t_vs_Iapp(sim.a_Results.t-rt,I,ax[2])
    if I_i!=[]:
        ax[2].set_ylim(I_i)
    plt.grid(b=False)
    ax[0].tick_params(labelsize=size_axisnumbers_font)
    ax[1].tick_params(labelsize=size_axisnumbers_font)
    ax[2].tick_params(labelsize=size_axisnumbers_font)
    locatory0 = MaxNLocator(nbins=4) # with 3 bins you will have 4 ticks
    ax[0].yaxis.set_major_locator(locatory0)
    locatory1 = MaxNLocator(nbins=4) # with 3 bins you will have 4 ticks
    ax[1].yaxis.set_major_locator(locatory1)
    locatory2 = MaxNLocator(nbins=4) # with 3 bins you will have 4 ticks
    ax[2].yaxis.set_major_locator(locatory2)
    if ms2s!=[]:
        ax[2].set_xlabel('time[s]')
    if t_i==[]:
        plt.autoscale(enable=True, axis='x', tight=True)
    else:
        plt.xlim(t_i)
    plt.grid(b=False)
    return f1

def plot_oneSim_Basic(sim,optional_title=[],t_i=[],reset_t=[]):
    if reset_t!=[]:
        rt=reset_t
    else:
        rt=0
    f1, ax= plt.subplots(3, sharex=True, facecolor="1",figsize=(fig_wide,fig_height))
    ax[0]=plot_t_vs_V(sim,ax[0],reset_t=reset_t)
    if optional_title!=[]:
        ax[0].set_title(optional_title)
    ax[1]=plot_t_vs_Concentrations(sim,ax[1],reset_t=reset_t)
    exec(sim.d_Protocol['s_Executable_stimulus'])
    I=[I_exp1(i)  for i in sim.a_Results.t]
    ax[2]=plot_t_vs_Iapp(sim.a_Results.t-rt,I,ax[2])
    ax[0].tick_params(labelsize=size_axisnumbers_font)
    ax[1].tick_params(labelsize=size_axisnumbers_font)
    ax[2].tick_params(labelsize=size_axisnumbers_font)
    plt.autoscale(enable=True, axis='x', tight=True)
    if t_i==[]:
        plt.autoscale(enable=True, axis='x', tight=True)
    else:
        plt.xlim(t_i)
    return f1

def create_orbit_specf_dic00(INFO,i,j,k,i_ss='n_K',j_ss='v',dir_figs=[],scale_i_vects=0.1,scale_j_vects=120*0.1,plt_Jvect=False,mark_currs=False,axi=[],i_api=[]):
    from numpy import linalg
    iiv=[]
    for i_s,j_s in INFO.items():
        if 'Jac_info' in i_s:
            iiv.append(int(i_s[i_s.index('o')+1:]))

    I_ap_v=[]
    for ii in iiv:
        i_ap=INFO['node_pars'+str(ii)]['I_app']
        if i_ap in I_ap_v:
            pass
        else:
            coin_iap=[]
            for i_ap_comp in I_ap_v:
                if abs(i_ap-i_ap_comp)<0.1:
                    coin_iap=1
            if coin_iap==[]:
                I_ap_v.append(i_ap)
    v_figs=[]
    v_iapp=[]
    v_ina=[]
    v_iko=[]
    v_iki=[]
    for i_ap in I_ap_v:
        fig, ax= plt.subplots(1, frameon=False, facecolor="1",figsize=(15,15))
        for ii in iiv:
            one_lc_iapp=[]
            if abs(i_ap-INFO['node_pars'+str(ii)]['I_app'])<0.1:
                saddle_point=INFO['Jac_info'+str(ii)][0]
                m_Jmat=INFO['Jac_info'+str(ii)][1]
                Jeival=INFO['Jac_info'+str(ii)][2]
                Jeivect=INFO['Jac_info'+str(ii)][3]
                try:
                    c_lc_iap=0
                    for ilc_iap in INFO['One_lc_iapp']:
                        if abs(i_ap-ilc_iap)<0.1:
                            one_lc_iapp=c_lc_iap
                        c_lc_iap+=1
                except:
                    pass
                if one_lc_iapp !=[]:
                    one_lc=INFO['One_lc'][one_lc_iapp]
                    one_lc_s=INFO['One_lc_s'][one_lc_iapp]
                else:
                    one_lc=[]
                    one_lc_s=[]
                # J=sympify(INFO['s_Jac_fun'])
                # Jfoo = lambdify(INFO['One_lc_s'],J)
                Jvects=[]
                widths=[]
                if k!=[]:
                    for vect in Jeivect:
                        vv=[saddle_point[i][0],saddle_point[j][0],saddle_point[k][0],vect[i].real,0,0]
                        Jvects.append(vv)
                        vv=[saddle_point[i][0],saddle_point[j][0],saddle_point[k][0],0,vect[j].real,0]
                        Jvects.append(vv)
                        vv=[saddle_point[i][0],saddle_point[j][0],saddle_point[k][0],0,0,vect[k].real]
                        Jvects.append(vv)
                    X,Y,Z,W,A,B=zip(*Jvects)
                if k==[]:
                    if plt_Jvect:
                        for i_pos in [0,1]:
                            vvv0=[INFO['Jeivects'+str(ii)][i]['Vect'][i_pos][i,i].real,INFO['Jeivects'+str(ii)][i]['Vect'][i_pos][i,j].real]
                            widths.append(copy(linalg.norm(vvv0)))
                            vvv1=vvv0/linalg.norm(vvv0)
                            if i_pos==0:
                                p=[saddle_point[i][0],saddle_point[j][0]]
                            else:
                                p=[saddle_point[i][0]-vvv1[0]*scale_i_vects,saddle_point[j][0]-vvv1[1]*scale_j_vects]
                            vvv1[0]=vvv1[0]*scale_i_vects
                            vvv1[1]=vvv1[1]*scale_j_vects
                            vv=p+vvv1.tolist()
                            Jvects.append(copy(vv))
                            vvv0=[INFO['Jeivects'+str(ii)][j]['Vect'][i_pos][j,i].real,INFO['Jeivects'+str(ii)][j]['Vect'][i_pos][j,j].real]
                            widths.append(linalg.norm(vvv0))
                            vvv1=vvv0/linalg.norm(vvv0)
                            if i_pos==0:
                                p=[saddle_point[i][0],saddle_point[j][0]]
                            else:
                                p=[saddle_point[i][0]-vvv1[0]*scale_i_vects,saddle_point[j][0]-vvv1[1]*scale_j_vects]
                            vvv1[0]=vvv1[0]*scale_i_vects
                            vvv1[1]=vvv1[1]*scale_j_vects
                            vv=p+vvv1.tolist()
                            Jvects.append(copy(vv))
                        for vect in Jeivect:
                            vv=[saddle_point[i][0],saddle_point[j][0],vect[i],vect[j]]
                            Jvects.append(vv)
                        X,Y,Z,W=zip(*Jvects)
                if k!=[]:
                    try:
                        ax.plot(one_lc[i],one_lc[j],one_lc[k],'o',color=[0,0,0])
                    except:
                        pass
                    # ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=20)
                    if any(Jeival.real>0):
                        ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=10,mfc='none')
                    else:
                        ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=10)
                    # ax.quiver(X,Y,Z,W,A,B, length=0.1)
                    try:
                        ax.set_xlabel('$'+one_lc_s[i]+'$',fontsize=size_axis_font)
                        ax.set_ylabel('$'+one_lc_s[j]+'$',fontsize=size_axis_font)
                        ax.set_zlabel('$'+one_lc_s[k]+'$',fontsize=size_axis_font)
                    except:
                        pass
                if k==[]:
                    if one_lc!=[]:
                        ax.plot(one_lc[i],one_lc[j],color=[0,0,0],linewidth=3.0)
                        if axi!=[] and abs(i_api-i_ap)<0.1:
                            axi.plot(one_lc[i],one_lc[j],color=[0,0,0],linewidth=3.0)
                    if any(Jeival.real>0):
                        ax.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=15)
                        ax.plot(saddle_point[i],saddle_point[j],'o',color=[1,1,1],markersize=8)
                        if axi!=[] and abs(i_api-i_ap)<0.1:
                            axi.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=15)
                            axi.plot(saddle_point[i],saddle_point[j],'o',color=[1,1,1],markersize=8)
                    else:
                        ax.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=15)
                        if axi!=[] and abs(i_api-i_ap)<0.1:
                            axi.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=15)
                        str_snpos=[saddle_point[i],saddle_point[j]]
                    # if one_lc_s!=[]:
                    #     ax.set_xlabel('$'+one_lc_s[i]+'$',fontsize=size_axis_font)
                    #     ax.set_ylabel('$'+one_lc_s[j]+'$',fontsize=size_axis_font)
                    # else:
                    #     ax.set_xlabel('$'+i_ss+'$',fontsize=size_axis_font)
                    #     ax.set_ylabel('$'+j_ss+'$',fontsize=size_axis_font)
                    # ax.set_title('$Iapp=$'+'{0:.2g}'.format(i_ap))
                    if plt_Jvect:
                        for j_vect in Jvects:
                            ax.annotate('', xy = (j_vect[0], j_vect[1]),xytext = (j_vect[2]+j_vect[0], j_vect[3]+j_vect[1]),arrowprops=dict(facecolor='black', shrink=0.05))
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-500,80])
        ax.axis('off')
        if dir_figs!=[]:
            name_file='Orbit_snap_in_Bif_Diagram_Nai_'+str(int(INFO['node_pars'+str(ii)]['Na_i']))+'_Ki_'+str(int(INFO['node_pars'+str(ii)]['K_i']))+'_Ko_'+str(int(INFO['node_pars'+str(ii)]['K_o']))+'_Iapp_'+str(int(i_ap))
            md_dir={
             'Title':'Frozen Orbit graph',
             'Author':"Susana Contreras 24/01/2018",
             'Subject':'Bifuration Diagram Nai='+str(INFO['node_pars'+str(ii)]['Na_i'])+' Ki='+str(INFO['node_pars'+str(ii)]['K_i'])+' Iapp='+str(i_ap),
             'Keywords':"Na-K-Pump, dependent excitability, adaptation, Bif analysis"
             }
            fig_file_title=name_file
            saving_pdf_figure(fig,dir_figs+fig_file_title,md_dir,pickle_f=False)
            v_figs.append(dir_figs+fig_file_title)

        if dir_figs==[]:
            # v_figs.append(copy(ax))
            if i_api!=[] and axi!=[]:
                v_figs=axi
                plt.close("fig")
        v_iapp.append(i_ap)
        v_ina.append(INFO['node_pars'+str(ii)]['Na_i'])
        v_iki.append(INFO['node_pars'+str(ii)]['K_i'])
        v_iko.append(INFO['node_pars'+str(ii)]['K_o'])
    return v_figs,v_iapp,v_ina,v_iko,v_iki

def create_orbit_figure(i,j,k,name_file,fig_file_title,dir_figs,scale_i_vects=0.1,scale_j_vects=120*0.1,plt_Jvect=False,mark_currs=False):
    f = open(dir_data+name_file+'.pk1','rb')
    INFO=pickle.load(f)
    neuron_snap=generic_neuron_from_json(INFO['neuron_model'],dir_file=INFO['neuron_dir'])
    fig, ax= plt.subplots(1, sharex=True, facecolor="1",figsize=(30,30))
    iiv=[]
    for i_s,j_s in INFO.items():
        if 'Jac_info' in i_s:
            iiv.append(int(i_s[i_s.index('o')+1:]))
    bifpar=INFO['neuron_pars']
    autobifpart=INFO['neuron_pars_auto']
    x_neuron=copy(neuron_snap)
    x_neuron.changing_pars(bifpar)
    s_membpot=x_neuron.s_resting_membrane_potentials
    v_membpot=x_neuron.resting_membrane_potentials(x_neuron.current_state)
    s_curr=x_neuron.s_curr
    c=0
    for i_s in s_curr:
        if i_s=='i_p':
            c_ip=c
        c+=1
    v_curr=x_neuron.neuron_currents(x_neuron.current_state)
    n_ip=v_curr[c_ip]
    if k==[]:
        c=0
        for i_s in s_membpot:
            i_v=scipy.linspace(min(INFO['One_lc'][i]),max(INFO['One_lc'][i]),10)
            ax.plot(i_v,np.ones(len(i_v))*v_membpot[c],linewidth=3,label='$'+i_s+'$')
            c+=1
    else:
        ax = f.gca(projection='3d')
        c=0
        for i_s in s_membpot:
            i_v=scipy.linspace(min(INFO['One_lc'][i]),max(INFO['One_lc'][i]),10)
            k_v=scipy.linspace(min(INFO['One_lc'][k]),max(INFO['One_lc'][k]),10)
            ax.plot(i_v,np.ones(len(i_v))*v_membpot[c],k_v,label='$'+i_s+'$')
            c+=1
    str_snpos=None
    for ii in iiv:
        saddle_point=INFO['Jac_info'+str(ii)][0]
        m_Jmat=INFO['Jac_info'+str(ii)][1]
        Jeival=INFO['Jac_info'+str(ii)][2]
        Jeivect=INFO['Jac_info'+str(ii)][3]
        one_lc=INFO['One_lc']
        one_lc_s=INFO['One_lc_s']
        # J=sympify(INFO['s_Jac_fun'])
        # Jfoo = lambdify(INFO['One_lc_s'],J)
        Jvects=[]
        widths=[]
        if k!=[]:
            for vect in Jeivect:
                vv=[saddle_point[i][0],saddle_point[j][0],saddle_point[k][0],vect[i].real,0,0]
                Jvects.append(vv)
                vv=[saddle_point[i][0],saddle_point[j][0],saddle_point[k][0],0,vect[j].real,0]
                Jvects.append(vv)
                vv=[saddle_point[i][0],saddle_point[j][0],saddle_point[k][0],0,0,vect[k].real]
                Jvects.append(vv)
            X,Y,Z,W,A,B=zip(*Jvects)
        if k==[]:
            for i_pos in [0,1]:
                vvv0=[INFO['Jeivects'+str(ii)][i]['Vect'][i_pos][i,i].real,INFO['Jeivects'+str(ii)][i]['Vect'][i_pos][i,j].real]
                widths.append(copy(linalg.norm(vvv0)))
                vvv1=vvv0/linalg.norm(vvv0)
                if i_pos==0:
                    p=[saddle_point[i][0],saddle_point[j][0]]
                else:
                    p=[saddle_point[i][0]-vvv1[0]*scale_i_vects,saddle_point[j][0]-vvv1[1]*scale_j_vects]
                vvv1[0]=vvv1[0]*scale_i_vects
                vvv1[1]=vvv1[1]*scale_j_vects
                vv=p+vvv1.tolist()
                Jvects.append(copy(vv))
                vvv0=[INFO['Jeivects'+str(ii)][j]['Vect'][i_pos][j,i].real,INFO['Jeivects'+str(ii)][j]['Vect'][i_pos][j,j].real]
                widths.append(linalg.norm(vvv0))
                vvv1=vvv0/linalg.norm(vvv0)
                if i_pos==0:
                    p=[saddle_point[i][0],saddle_point[j][0]]
                else:
                    p=[saddle_point[i][0]-vvv1[0]*scale_i_vects,saddle_point[j][0]-vvv1[1]*scale_j_vects]
                vvv1[0]=vvv1[0]*scale_i_vects
                vvv1[1]=vvv1[1]*scale_j_vects
                vv=p+vvv1.tolist()
                Jvects.append(copy(vv))
            for vect in Jeivect:
                vv=[saddle_point[i][0],saddle_point[j][0],vect[i],vect[j]]
                Jvects.append(vv)
            X,Y,Z,W=zip(*Jvects)
        if k!=[]:
            ax.plot(INFO['One_lc'][i],INFO['One_lc'][j],INFO['One_lc'][k],'o',color=[0,0,0])
            # ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=20)
            if any(Jeival.real>0):
                ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=10,mfc='none')
            else:
                ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=10)
            # ax.quiver(X,Y,Z,W,A,B, length=0.1)
            ax.set_xlabel('$'+one_lc_s[i]+'$',fontsize=size_axis_font)
            ax.set_ylabel('$'+one_lc_s[j]+'$',fontsize=size_axis_font)
            ax.set_zlabel('$'+one_lc_s[k]+'$',fontsize=size_axis_font)
        else:
            ax.plot(INFO['One_lc'][i],INFO['One_lc'][j],color=[0,0,0])
            if any(Jeival.real>0):
                ax.plot(saddle_point[i],saddle_point[j],'o',markersize=10,mfc='none')
            else:
                ax.plot(saddle_point[i],saddle_point[j],'o',markersize=10)
                v_curr=x_neuron.neuron_currents(saddle_point)
                str_sn=''.join([jj+'['+'{0:.2g}'.format(float(v_curr[ii]))+']' for ii,jj in enumerate(s_curr)])
                str_snpos=[saddle_point[i],saddle_point[j]]
            ax.set_xlabel('$'+one_lc_s[i]+'$',fontsize=size_axis_font)
            ax.set_ylabel('$'+one_lc_s[j]+'$',fontsize=size_axis_font)
            ax.set_title('Orbit $E_K=$'+'{0:.2g}'.format(int(x_neuron.resting_membrane_potentials(x_neuron.current_state)[0])))
            if plt_Jvect:
                for j_vect in Jvects:
                    ax.annotate('', xy = (j_vect[0], j_vect[1]),xytext = (j_vect[2]+j_vect[0], j_vect[3]+j_vect[1]),arrowprops=dict(facecolor='black', shrink=0.05))
    if mark_currs:
        j_vect=Jvects[-1]
        try:
            ax.annotate('$I_app['+'{0:.2g}'.format(INFO['node_pars'+str(ii)]['I_app'])+']- i_p['+'{0:.2g}'.format(n_ip)+']='+'{0:.2g}'.format(INFO['node_pars'+str(ii)]['I_app']-n_ip)+'$', xy = (0, 59),xytext = (-0.05, 60),arrowprops=dict(facecolor='black', shrink=0.05))
        except:
            pass
        a,b=shape(INFO['One_lc'])
        vv=[]
        i_app_v=[]
        for i_r in range(b):
            v_curr_i=x_neuron.neuron_currents(np.array(np.matrix(INFO['One_lc'])[:,i_r]))
            vv.append(v_curr_i)
            try:
                i_app_v.append(INFO['node_pars'+str(ii)]['I_app'])
            except:
                pass
        v_sum_curr=sum(vv,axis=0)
        str_lc=''.join([jj+'['+'{0:.2g}'.format(float(v_sum_curr[ii]))+']' for ii,jj in enumerate(s_curr)])
        pos_lc_m=[INFO['One_lc'][i][argmin(INFO['One_lc'][j])],INFO['One_lc'][j][argmin(INFO['One_lc'][j])]]
        ax.annotate('$'+str_lc+' I_{app}'+'['+'{0:.2g}'.format(sum(i_app_v))+']='+'{0:.2g}'.format(float(sum(i_app_v)-sum(v_sum_curr)))+'$', xy = (pos_lc_m[0], pos_lc_m[1]),xytext = (-0.1, pos_lc_m[1]*0.9),arrowprops=dict(facecolor='black', shrink=0.05))
        # ax.annotate('$='+'{0:.2g}'.format(float(sum(i_app_v)-sum(v_sum_curr)))+'$', xy = (pos_lc_m[0], pos_lc_m[1]),xytext = (pos_lc_m[0]*0.9, pos_lc_m[1]*1.1),arrowprops=dict(facecolor='black', shrink=0.05))
        if str_snpos:
            try:
                ax.annotate('$'+str_sn+'I_{app}'+'['+'{0:.2g}'.format(INFO['node_pars'+str(ii)]['I_app'])+']='+'{0:.2g}'.format(INFO['node_pars'+str(ii)]['I_app']-float(sum(v_curr)))+' $', xy = (str_snpos[0], str_snpos[1]),xytext = (-0.1, str_snpos[1]*1.1),arrowprops=dict(facecolor='black', shrink=0.05))
                # ax.annotate('$='+'{0:.2g}'.format(sum(INFO['node_pars'+str(ii)]['I_app'])-sum(v_curr))+' $', xy = (str_snpos[0], str_snpos[1]),xytext = (-0.1, str_snpos[1]*1.4),arrowprops=dict(facecolor='black', shrink=0.05))
            except:
                pass
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-100,80])

    return fig

def zoomingBox(ax1, roi, ax2, color='red', linewidth=2, roiKwargs={}, arrowKwargs={}):
    from matplotlib.patches import Rectangle
    '''
    **Notes (for reasons unknown to me)**
    1. Sometimes the zorder of the axes need to be adjusted manually...
    2. The figure fraction is accurate only with qt backend but not inline...
    '''
    roiKwargs = dict([('fill',False), ('linestyle','dashed'), ('color',color), ('linewidth',linewidth)] + roiKwargs.items())
    ax1.add_patch(Rectangle([roi[0],roi[2]], roi[1]-roi[0], roi[3]-roi[2], **roiKwargs))
    arrowKwargs = dict([('arrowstyle','-'), ('color',color), ('linewidth',linewidth)] + arrowKwargs.items())
    srcCorners = [[roi[0],roi[2]], [roi[0],roi[3]], [roi[1],roi[2]], [roi[1],roi[3]]]
    dstCorners = ax2.get_position().corners()
    srcBB = ax1.get_position()
    dstBB = ax2.get_position()
    if (dstBB.min[0]>srcBB.max[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.max[0]<srcBB.min[0] and dstBB.min[1]>srcBB.max[1]):
        src = [0, 3]; dst = [0, 3]
    elif (dstBB.max[0]<srcBB.min[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.min[0]>srcBB.max[0] and dstBB.min[1]>srcBB.max[1]):
        src = [1, 2]; dst = [1, 2]
    elif dstBB.max[1] < srcBB.min[1]:
        src = [0, 2]; dst = [1, 3]
    elif dstBB.min[1] > srcBB.max[1]:
        src = [1, 3]; dst = [0, 2]
    elif dstBB.max[0] < srcBB.min[0]:
        src = [0, 1]; dst = [2, 3]
    elif dstBB.min[0] > srcBB.max[0]:
        src = [2, 3]; dst = [0, 1]
    for k in range(2):
        ax1.annotate('', xy=dstCorners[dst[k]], xytext=srcCorners[src[k]], xycoords='figure fraction', textcoords='data', arrowprops=arrowKwargs)

def plot_whatever_vs_whatever2_picker(simx,simy,s_whateverX,s_whateverY,axarr,s_Optional_x="",s_Optional_y="",s_Optional_z="",s_Optional_title="",si=55,alphas=[]):
    markers = ['o','^','*','v']
    axarr.set_title(r' '+s_Optional_title,fontsize=size_title_font)
    axarr.set_ylabel(r' '+s_Optional_y,fontsize=size_axis_font)
    axarr.set_xlabel(r' '+s_Optional_x,fontsize=size_axis_font)
    axarr.grid(color='k', alpha=0.1, linestyle='solid', linewidth=0.5)
    axarr.tick_params(labelsize=size_axisnumbers_font)

    if len(s_whateverX)<len(s_whateverY):
        co=0
        for i in s_whateverY:
            if alphas==[]:
                b,=axarr.scatter(simx,simy[co], s=si, vmin=min(np.concatenate(simz)), vmax=max(np.concatenate(simz)), picker=5)
            else:
                b,=axarr.scatter(simx,simy[co],alpha=alphas[co], s=si, picker=5)
            # b,=axarr.scatter(simx,simy[co],c=simz[co], s=si, vmin=min(np.concatenate(simz)), vmax=max(np.concatenate(simz)), picker=5)
            co+=1
    if len(s_whateverX)==len(s_whateverY):
        co=0
        for i in s_whateverY:
            if alphas==[]:
                b,=axarr.scatter(concatenate(simx),concatenate(simy), s=si, picker=5)
            else:
                b=[]
                for j in alphas:
                    c=axarr.scatter(simx[co],simy[co],alpha=alphas[co], s=si, picker=5)
                    b.append(c)
                    c=[]
            co+=1

    cbar=plt.colorbar(b[0])
    cbar.set_label(s_Optional_z,size=size_axis_font)
    cbar.ax.tick_params(labelsize=size_axisnumbers_font)
    return axarr, b

def predefplot_inside_predefplot(fig,axx,f,kwargs,axinf,lims2zoom,pw="30%",ph="30%",locat=4,s_ylabel=[],s_xlabel=[],little_title=[],pointorline=1):
    ####### This function plots a plot inside a plot. Useful when there is a parameter exploration, and one needs to see how a dot on the parameter explorations looks like..
    ####### for example how the voltage trace of a specific parameter thing looks...
    ####### Inputs: fig : (big graph)
    #######         axx : (axes of the big graph)
    #######         f : (ploting function for the inner function)
    #######         kwargs : (inputs of the ploting function)
    #######         axinf : (position of the parent axes to input f)
    #######         lims2zoom : the area where the specific simulation lies down (data coordinates, [xmin,xmax,ymin,ymax]) (to draw rectangle on top of the big figure)
    #######         pw : Percentage width of the little figure with respect to the big one
    #######         ph : Percentage height of the little figure with respect to the big one
    #######         locat : location of the little figure on the graph
    #######         s_ylabel: optinal y labelfor the little graph
    #######         s_xlabel : optinal x labelfor the little graph
    #######         little title : optinal title for the little graph
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    x1, x2, y1, y2 = lims2zoom[0], lims2zoom[1], lims2zoom[2], lims2zoom[3]
    # fig, ax= plt.subplots(1, sharex=True, facecolor="1",figsize=(10,12))
    ax=axx

    arx=ax.get_xlim()
    ary=ax.get_ylim()
    if locat==4:
        drawArrow([(arx[1]-arx[0])*0.95+arx[0],(ary[1]-ary[0])*0.05+ary[0]],[x1+(x2-x1)/2,(y2-y1)/2+y1],ax)
    if locat==3:
        drawArrow([(arx[1]-arx[0])*0.05+arx[0],(ary[1]-ary[0])*0.05+ary[0]],[x1+(x2-x1)/2,(y2-y1)/2+y1],ax)
    if locat==2:
        drawArrow([(arx[1]-arx[0])*0.05+arx[0],(ary[1]-ary[0])*0.95+ary[0]],[x1+(x2-x1)/2,(y2-y1)/2+y1],ax)
    if locat==1:
        drawArrow([(arx[1]-arx[0])*0.95+arx[0],(ary[1]-ary[0])*0.95+ary[0]],[x1+(x2-x1)/2,(y2-y1)/2+y1],ax)

    axins = inset_axes(ax,
                        width=pw, # width = 30% of parent_bbox
                        height=ph, # height : 1 inch
                        loc=locat)

    #axins[0]=axx2
    #f(kwargs[0],kwargs[1],kwargs[2],kwargs[3],axins,kwargs[5],kwargs[6])
    lst=list(kwargs)
    lst[axinf]=axins
    #kwargs2=kwargs[0:3],axins,kwargs[5:7]
    kwargs2=tuple(lst)
    f(*kwargs2)
    #axins.scatter(x,y)
    transFigure = fig.transFigure.inverted()
    coord_ax1 = transFigure.transform(ax.transData.transform([x1,y1]))
    coord_ax2 = transFigure.transform(ax.transData.transform([x2,y2]))

    ax2=fig.add_axes([coord_ax1[0], coord_ax1[1], coord_ax2[0]-coord_ax1[0], coord_ax2[1]-coord_ax1[1]])
    ax2.patch.set_alpha(0.1)
    ax2.set_xticks([])
    ax2.set_yticks([])

    return fig

def draw_manifolds_wPhasePlane(v_coords,name_file,vis_easy=False):
    f = open(dir_data+name_file+'.pk1','rb')
    INFO=pickle.load(f)

    if len(v_coords)>2:
        fig = plt.figure(facecolor="1",figsize=(10,10))
        ax = fig.gca(projection='3d')
    else:
        fig, ax= plt.subplots(1, sharex=True, facecolor="1",figsize=(10,10))

    iiv=[]
    v_s0=[]
    for i_s,j_s in INFO.items():
        if 'Jac_info' in i_s:
            cc=int(i_s[i_s.index('o')+1:])
            if any([abs(Vv-INFO['Jac_info'+str(cc)][0][0])<0.1 for Vv in v_s0]):
                pass
            else:
                iiv.append(int(i_s[i_s.index('o')+1:]))
                v_s0.append(INFO['Jac_info'+str(cc)][0][0])

    saddle_point_v=[]
    Jeival_v=[]
    Jeivect_v=[]
    one_lc=[]
    one_lc=INFO['One_lc']
    one_lc_s=INFO['One_lc_s']
    vm_v=[]
    pos_v=[]
    from cmath import sqrt
    from mpl_toolkits.mplot3d import Axes3D
    for ii in iiv:
        saddle_point=INFO['Jac_info'+str(ii)][0]
        m_Jmat=INFO['Jac_info'+str(ii)][1]
        Jeival=INFO['Jac_info'+str(ii)][2]
        Jeivect=INFO['Jac_info'+str(ii)][3]
        vv=norm(Jeival[v_coords])
        # scale=norm(Jeival[v_coords])
        # vv=1
        scale_manifold=0.1
        hwarrow=0.8
        hlarrow=0.01
        for jj in v_coords:
            if max(one_lc[v_coords[0]])<=1:
                dxx=scale_manifold*1.0
                ax.set_xlim(-0.1,1.1)

            else:
                dxx=scale_manifold*180.0
                ax.set_xlim(-100,80)

            if max(one_lc[v_coords[1]])<=1:
                dyy=scale_manifold*1.0
                ax.set_ylim(-0.1,1.1)

            else:
                dyy=scale_manifold*180.0
                ax.set_ylim(-100,80)
        c=0

        mJeival=mean(abs(Jeival.real))
        n_arrows=0
        n_arrows0=-0.5
        for i_j in np.argsort(abs(Jeival)):
        # for i_j in range(len(saddle_point)):
            n_arrows0+=0.5
            n_arrows=1+int(n_arrows0)
            # print('Jeival='+str(Jeival[i_j].real)+'arrows'+str(n_arrows))
            if c==0:
                dx=dxx
                dy=0
            else:
                dx=0
                dy=dyy
            if Jeival[i_j].real>0:
                if len(v_coords)==2:
                    # ax.plot(xx,yycalc ,color='red')
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    #### try2
                    y1 = [saddle_point[v_coords[0]]]
                    y2 = [saddle_point[v_coords[1]]]
                    Y1, Y2 = np.meshgrid(y1, y2)
                    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
                    NI, NJ = Y1.shape
                    for i in range(NI):
                        for j in range(NJ):
                            x = Y1[i, j]
                            y = Y2[i, j]
                            # u[i,j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j][v_coords].real)
                            # v[i,j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j][v_coords].real)
                            u[i,j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j].real)
                            v[i,j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j].real)
                            if vis_easy:
                                u[i,j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j][v_coords].real)
                                v[i,j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j][v_coords].real)
                    # print('Un-Stable'+ str(u**2+v**2))
                    # Q = ax.quiver(Y1, Y2, u, v, color='r')
                    if n_arrows==6:
                        for cc_arrows in range(n_arrows):
                            print('pos '+str(saddle_point)+' Jeival'+str(Jeival[i_j])+' ['+str(u)+str(v)+']')
                            ax.annotate('', (u+saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), v+saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                #xycoords="figure fraction", textcoords="figure fraction",
                                ha="right", va="center",
                                size=size_axis_font,
                                arrowprops=dict(arrowstyle='->',
                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                lw=2.0,
                                                # patchB=p,
                                                shrinkA=5,
                                                shrinkB=5,
                                                fc="k", ec="k",
                                                connectionstyle="arc3,rad=-0.0",
                                                ))
                            ax.annotate('',(-u+saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), -v+saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                #xycoords="figure fraction", textcoords="figure fraction",
                                ha="right", va="center",
                                size=size_axis_font,
                                arrowprops=dict(arrowstyle='->',
                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                lw=2.0,
                                                # patchB=p,
                                                shrinkA=5,
                                                shrinkB=5,
                                                fc="k", ec="k",
                                                connectionstyle="arc3,rad=-0.0",
                                                ))
                if len(v_coords)==3:
                    scale=norm(Jeival[i_j])/norm(Jeival)
                    a=Jeivect[i_j][v_coords[0]].real/norm(Jeivect[v_coords])
                    b=Jeivect[i_j][v_coords[1]].real/norm(Jeivect[v_coords])
                    d=Jeivect[i_j][v_coords[2]].real/norm(Jeivect[v_coords])
                    c=-d*saddle_point[v_coords[2]]-b*saddle_point[v_coords[1]]-a*saddle_point[v_coords[0]]
                    X,Y = np.meshgrid(np.arange(saddle_point[v_coords[0]]-dxx, saddle_point[v_coords[0]]+dxx, dxx/2), np.arange(saddle_point[v_coords[1]]-dyy, saddle_point[v_coords[1]]+dyy, dyy/2))
                    XX = X.flatten()
                    YY = Y.flatten()
                    Z = (-a*X - b*Y - c)/d
                    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,color='red')
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    ax.set_zlabel('$'+one_lc_s[v_coords[2]]+'$',fontsize=size_axis_font)
                    if one_lc_s[v_coords[2]]=='v':
                        ax.set_zlim(-100, 80)
                    else:
                        ax.set_zlim(0, 1)

            if Jeival[i_j].real<0:
                if len(v_coords)==2:
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    #### try2
                    y1 = [saddle_point[v_coords[0]]]
                    y2 = [saddle_point[v_coords[1]]]
                    Y1, Y2 = np.meshgrid(y1, y2)
                    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
                    NI, NJ = Y1.shape
                    for i in range(NI):
                        for j in range(NJ):
                            x = Y1[i, j]
                            y = Y2[i, j]
                            u[i,j] = dxx*Jeivect[i_j][v_coords[0]].real/norm(Jeivect[i_j].real)
                            v[i,j] = dyy*Jeivect[i_j][v_coords[1]].real/norm(Jeivect[i_j].real)
                            if vis_easy:
                                u[i,j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j][v_coords].real)
                                v[i,j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j][v_coords].real)
                    # print('Stable '+ str(u**2+v**2))
                    # print('pos '+str(saddle_point)+' Jeival'+str(Jeival[i_j])+' ['+str(u)+str(v)+']')
                    if n_arrows==1:
                        for cc_arrows in range(n_arrows):
                            ax.annotate('', (saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))), (u+saddle_point[v_coords[0]], v+saddle_point[v_coords[1]]),
                                ha="right", va="center",
                                size=size_axis_font,
                                arrowprops=dict(arrowstyle='->',
                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                lw=2.0,
                                                # patchB=p,
                                                shrinkA=5,
                                                shrinkB=5,
                                                fc="k", ec="k",
                                                connectionstyle="arc3,rad=-0.0",
                                                ))
                            ax.annotate('', (saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))), (-u+saddle_point[v_coords[0]], -v+saddle_point[v_coords[1]]),
                                ha="right", va="center",
                                size=size_axis_font,
                                arrowprops=dict(arrowstyle='->',
                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                lw=2.0,
                                                # patchB=p,
                                                shrinkA=5,
                                                shrinkB=5,
                                                fc="k", ec="k",
                                                connectionstyle="arc3,rad=0.0",
                                                ))
                if len(v_coords)==3:
                    scale=norm(Jeival[i_j])/norm(Jeival)
                    a=Jeivect[i_j][v_coords[0]].real/norm(Jeivect[v_coords])
                    b=Jeivect[i_j][v_coords[1]].real/norm(Jeivect[v_coords])
                    d=Jeivect[i_j][v_coords[2]].real/norm(Jeivect[v_coords])
                    c=-d*saddle_point[v_coords[2]]-b*saddle_point[v_coords[1]]-a*saddle_point[v_coords[0]]
                    X,Y = np.meshgrid(np.arange(saddle_point[v_coords[0]]-dxx, saddle_point[v_coords[0]]+dxx, dxx/2), np.arange(saddle_point[v_coords[1]]-dyy, saddle_point[v_coords[1]]+dyy, dyy/2))
                    XX = X.flatten()
                    YY = Y.flatten()
                    Z = (-a*X - b*Y - c)/d
                    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,color='blue')
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    ax.set_zlabel('$'+one_lc_s[v_coords[2]]+'$',fontsize=size_axis_font)
                    if one_lc_s[v_coords[2]]=='v':
                        ax.set_zlim(-100, 80)
                    else:
                        ax.set_zlim(0, 1)

            c+=1
        if len(v_coords)==2:
            ax.plot(one_lc[v_coords[0]],one_lc[v_coords[1]],color=[0,0,0])
            if any(Jeival.real>0):
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],'o',markersize=10,mfc='none')
            else:
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],'o',markersize=10)
                print(ii)
        if len(v_coords)==3:
            ax.plot(one_lc[v_coords[0]],one_lc[v_coords[1]],one_lc[v_coords[2]],color=[0,0,0])
            if any(Jeival.real>0):
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],saddle_point[v_coords[2]],'o',markersize=10,mfc='none')
            else:
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],saddle_point[v_coords[2]],'o',markersize=10)
                print(ii)
    return(fig)

def draw_manifolds_wPhasePlane_and_L0(v_coords,name_file,vis_easy=False):
    f = open(dir_data+name_file+'.pk1','rb')
    INFO=pickle.load(f)

    if len(v_coords)>2:
        fig = plt.figure(facecolor="1",figsize=(10,10))
        ax = fig.gca(projection='3d')
    else:
        fig, ax= plt.subplots(1, sharex=True, facecolor="1",figsize=(10,10))

    iiv=[]
    v_s0=[]
    for i_s,j_s in INFO.items():
        if 'Jac_info' in i_s:
            cc=int(i_s[i_s.index('o')+1:])
            if any([abs(Vv-INFO['Jac_info'+str(cc)][0][0])<0.1 for Vv in v_s0]):
                pass
            else:
                iiv.append(int(i_s[i_s.index('o')+1:]))
                v_s0.append(INFO['Jac_info'+str(cc)][0][0])

    saddle_point_v=[]
    Jeival_v=[]
    Jeivect_v=[]
    one_lc=[]
    one_lc=INFO['One_lc']
    one_lc_s=INFO['One_lc_s']
    vm_v=[]
    pos_v=[]
    from cmath import sqrt
    from mpl_toolkits.mplot3d import Axes3D
    from sympy import S, symbols, lambdify
    from numpy import linalg as LA
    for ii in iiv:
        saddle_point=INFO['Jac_info'+str(ii)][0]
        m_Jmat=INFO['Jac_info'+str(ii)][1]
        Jeival=INFO['Jac_info'+str(ii)][2]
        Jeivect=INFO['Jac_info'+str(ii)][3]
        vv=norm(Jeival[v_coords])
        s_svars=symbols(INFO['One_lc_s'])
        fun_jac=lambdify(s_svars,INFO['s_Jac_fun'])
        Jmat = fun_jac(*saddle_point)
        m_JMat = [[0 for x in range(len(Jmat))] for y in range(len(Jmat))]
        for i in range(len(Jmat)):
            for j in range(len(Jmat)):
                if 'array' in str(type(Jmat[i][j])):
                    m_JMat[i][j]=Jmat[i][j]
                else:
                    m_JMat[i][j]=np.array([Jmat[i][j]])
        m_JMat = [np.concatenate(j) for j in m_JMat]
        m_Jmat=np.matrix(m_JMat)
        t_Jmat=m_Jmat.transpose()
        #### Finding left eigenvectors
        lJeival, lJeivect = LA.eig(t_Jmat)
        lJeivect=array(lJeivect)
        lJeival=array(lJeival)
        # scale=norm(Jeival[v_coords])
        # vv=1
        scale_manifold=0.1
        hwarrow=0.8
        hlarrow=0.01
        for jj in v_coords:
            if max(one_lc[v_coords[0]])<=1:
                dxx=scale_manifold*1.0
                ax.set_xlim(-0.1,1.1)

            else:
                dxx=scale_manifold*180.0
                ax.set_xlim(-100,80)

            if max(one_lc[v_coords[1]])<=1:
                dyy=scale_manifold*1.0
                ax.set_ylim(-0.1,1.1)

            else:
                dyy=scale_manifold*180.0
                ax.set_ylim(-100,80)
        c=0

        mJeival=mean(abs(Jeival.real))
        n_arrows=0
        n_arrows0=-0.5
        for i_j in np.argsort(abs(Jeival)):
        # for i_j in range(len(saddle_point)):
            n_arrows0+=0.5
            n_arrows=1+int(n_arrows0)
            # print('Jeival='+str(Jeival[i_j].real)+'arrows'+str(n_arrows))
            if c==0:
                dx=dxx
                dy=0
            else:
                dx=0
                dy=dyy
            if Jeival[i_j].real>0:
                if len(v_coords)==2:
                    # ax.plot(xx,yycalc ,color='red')
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    #### try2
                    y1 = [saddle_point[v_coords[0]]]
                    y2 = [saddle_point[v_coords[1]]]
                    Y1, Y2 = np.meshgrid(y1, y2)
                    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
                    NI, NJ = Y1.shape
                    for i in range(NI):
                        for j in range(NJ):
                            x = Y1[i, j]
                            y = Y2[i, j]
                            # u[i,j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j][v_coords].real)
                            # v[i,j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j][v_coords].real)
                            u[i,j] = dxx*lJeivect[i_j][v_coords[0]]/norm(lJeivect[i_j].real)
                            v[i,j] = dyy*lJeivect[i_j][v_coords[1]]/norm(lJeivect[i_j].real)
                            if vis_easy:
                                u[i,j] = dxx*lJeivect[i_j][v_coords[0]]/norm(lJeivect[i_j][v_coords].real)
                                v[i,j] = dyy*lJeivect[i_j][v_coords[1]]/norm(lJeivect[i_j][v_coords].real)
                    # print('Un-Stable'+ str(u**2+v**2))
                    # Q = ax.quiver(Y1, Y2, u, v, color='r')
                    # if n_arrows==6:
                    if sum(Jeival.real>0)==1:
                        for cc_arrows in range(n_arrows):
                                print('pos '+str(saddle_point)+' Jeival'+str(lJeival[i_j])+' ['+str(u)+str(v)+']')
                                ax.annotate('', (u+saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), v+saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                    #xycoords="figure fraction", textcoords="figure fraction",
                                    ha="right", va="center",
                                    size=size_axis_font,
                                    arrowprops=dict(arrowstyle='->',
                                                    # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                    lw=2.0,
                                                    # patchB=p,
                                                    shrinkA=5,
                                                    shrinkB=5,
                                                    fc="k", ec="k",
                                                    connectionstyle="arc3,rad=-0.0",
                                                    ))
                                ax.annotate('',(-u+saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), -v+saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                    #xycoords="figure fraction", textcoords="figure fraction",
                                    ha="right", va="center",
                                    size=size_axis_font,
                                    arrowprops=dict(arrowstyle='->',
                                                    # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                    lw=2.0,
                                                    # patchB=p,
                                                    shrinkA=5,
                                                    shrinkB=5,
                                                    fc="k", ec="k",
                                                    connectionstyle="arc3,rad=-0.0",
                                                    ))
                if len(v_coords)==3:
                    scale=norm(Jeival[i_j])/norm(Jeival)
                    a=Jeivect[i_j][v_coords[0]].real/norm(Jeivect[v_coords])
                    b=Jeivect[i_j][v_coords[1]].real/norm(Jeivect[v_coords])
                    d=Jeivect[i_j][v_coords[2]].real/norm(Jeivect[v_coords])
                    c=-d*saddle_point[v_coords[2]]-b*saddle_point[v_coords[1]]-a*saddle_point[v_coords[0]]
                    X,Y = np.meshgrid(np.arange(saddle_point[v_coords[0]]-dxx, saddle_point[v_coords[0]]+dxx, dxx/2), np.arange(saddle_point[v_coords[1]]-dyy, saddle_point[v_coords[1]]+dyy, dyy/2))
                    XX = X.flatten()
                    YY = Y.flatten()
                    Z = (-a*X - b*Y - c)/d
                    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,color='red')
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    ax.set_zlabel('$'+one_lc_s[v_coords[2]]+'$',fontsize=size_axis_font)
                    if one_lc_s[v_coords[2]]=='v':
                        ax.set_zlim(-100, 80)
                    else:
                        ax.set_zlim(0, 1)

            if Jeival[i_j].real<0:
                if len(v_coords)==2:
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    #### try2
                    y1 = [saddle_point[v_coords[0]]]
                    y2 = [saddle_point[v_coords[1]]]
                    Y1, Y2 = np.meshgrid(y1, y2)
                    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
                    NI, NJ = Y1.shape
                    for i in range(NI):
                        for j in range(NJ):
                            x = Y1[i, j]
                            y = Y2[i, j]
                            u[i,j] = dxx*Jeivect[i_j][v_coords[0]].real/norm(Jeivect[i_j].real)
                            v[i,j] = dyy*Jeivect[i_j][v_coords[1]].real/norm(Jeivect[i_j].real)
                            if vis_easy:
                                u[i,j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j][v_coords].real)
                                v[i,j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j][v_coords].real)
                    # print('Stable '+ str(u**2+v**2))
                    # print('pos '+str(saddle_point)+' Jeival'+str(Jeival[i_j])+' ['+str(u)+str(v)+']')
                    if n_arrows==1:
                        for cc_arrows in range(n_arrows):
                            pass
                            # ax.annotate('', (saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))), (u+saddle_point[v_coords[0]], v+saddle_point[v_coords[1]]),
                            #     ha="right", va="center",
                            #     size=size_axis_font,
                            #     arrowprops=dict(arrowstyle='->',
                            #                     # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                            #                     lw=2.0,
                            #                     # patchB=p,
                            #                     shrinkA=5,
                            #                     shrinkB=5,
                            #                     fc="k", ec="k",
                            #                     connectionstyle="arc3,rad=-0.0",
                            #                     ))
                            # ax.annotate('', (saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))), (-u+saddle_point[v_coords[0]], -v+saddle_point[v_coords[1]]),
                            #     ha="right", va="center",
                            #     size=size_axis_font,
                            #     arrowprops=dict(arrowstyle='->',
                            #                     # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                            #                     lw=2.0,
                            #                     # patchB=p,
                            #                     shrinkA=5,
                            #                     shrinkB=5,
                            #                     fc="k", ec="k",
                            #                     connectionstyle="arc3,rad=0.0",
                            #                     ))
                if len(v_coords)==3:
                    scale=norm(Jeival[i_j])/norm(Jeival)
                    a=Jeivect[i_j][v_coords[0]].real/norm(Jeivect[v_coords])
                    b=Jeivect[i_j][v_coords[1]].real/norm(Jeivect[v_coords])
                    d=Jeivect[i_j][v_coords[2]].real/norm(Jeivect[v_coords])
                    c=-d*saddle_point[v_coords[2]]-b*saddle_point[v_coords[1]]-a*saddle_point[v_coords[0]]
                    X,Y = np.meshgrid(np.arange(saddle_point[v_coords[0]]-dxx, saddle_point[v_coords[0]]+dxx, dxx/2), np.arange(saddle_point[v_coords[1]]-dyy, saddle_point[v_coords[1]]+dyy, dyy/2))
                    XX = X.flatten()
                    YY = Y.flatten()
                    Z = (-a*X - b*Y - c)/d
                    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,color='blue')
                    ax.set_xlabel('$'+one_lc_s[v_coords[0]]+'$',fontsize=size_axis_font)
                    ax.set_ylabel('$'+one_lc_s[v_coords[1]]+'$',fontsize=size_axis_font)
                    ax.set_zlabel('$'+one_lc_s[v_coords[2]]+'$',fontsize=size_axis_font)
                    if one_lc_s[v_coords[2]]=='v':
                        ax.set_zlim(-100, 80)
                    else:
                        ax.set_zlim(0, 1)

            c+=1
        if len(v_coords)==2:
            ax.plot(one_lc[v_coords[0]],one_lc[v_coords[1]],color=[0,0,0])
            if any(Jeival.real>0):
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],'o',markersize=10,mfc='none')
            else:
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],'o',markersize=10)
                print(ii)
        if len(v_coords)==3:
            ax.plot(one_lc[v_coords[0]],one_lc[v_coords[1]],one_lc[v_coords[2]],color=[0,0,0])
            if any(Jeival.real>0):
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],saddle_point[v_coords[2]],'o',markersize=10,mfc='none')
            else:
                ax.plot(saddle_point[v_coords[0]],saddle_point[v_coords[1]],saddle_point[v_coords[2]],'o',markersize=10)
                print(ii)
    return(fig)

# fig, ax= plt.subplots(1, sharex=True, facecolor="1",figsize=(10,10))
# ax.annotate('', (1, 1),
#     (1-0.5, 1-0.5),
#     #xycoords="figure fraction", textcoords="figure fraction",
#     ha="right", va="center",
#     size=size_axis_font,
#     arrowprops=dict(arrowstyle='->',
#                     shrinkA=5,
#                     shrinkB=5,
#                     fc="k", ec="k",
#                     connectionstyle="arc3,rad=-0.05",
#                     ))
#
# ax.annotate('', (1, 1),
#     (1+0.5, 1+0.5),
#     #xycoords="figure fraction", textcoords="figure fraction",
#     ha="right", va="center",
#     size=size_axis_font,
#     arrowprops=dict(arrowstyle='->',
#                     shrinkA=5,
#                     shrinkB=5,
#                     fc="k", ec="k",
#                     connectionstyle="arc3,rad=-0.05",
#                     ))
#
# ax.annotate('', (1, 1),
#     (-0.1, -0.1),
#     #xycoords="figure fraction", textcoords="figure fraction",
#     ha="right", va="center",
#     size=size_axis_font,
#     arrowprops=dict(arrowstyle='<-',
#                     shrinkA=5,
#                     shrinkB=5,
#                     fc="k", ec="k",
#                     connectionstyle="arc3,rad=-0.05",
#                     ))



def create_orbit_specf_dic(INFO,i,j,k,i_ss='n_K',j_ss='v',vis_easy=False,dir_figs=[],scale_i_vects=0.1,scale_j_vects=120*0.1,plt_Jvect=False,mark_currs=False,axi=[],i_api=[],Ei=[],small_marker=False):
    from numpy import linalg
    iiv=[]
    for i_s,j_s in INFO.items():
        if 'Jac_info' in i_s:
            iiv.append(int(i_s[i_s.index('o')+1:]))

    I_ap_v=[]
    for ii in iiv:
        i_ap=INFO['node_pars'+str(ii)]['I_app']
        if i_ap in I_ap_v:
            pass
        else:
            coin_iap=[]
            for i_ap_comp in I_ap_v:
                if abs(i_ap-i_ap_comp)<0.1:
                    coin_iap=1
            if coin_iap==[]:
                I_ap_v.append(i_ap)

    v_figs=[]
    v_iapp=[]
    v_ina=[]
    v_iko=[]
    v_iki=[]
    scale_manifold=0.1
    v_coords=[i,j]

    for i_ap in I_ap_v:
        fig, ax= plt.subplots(1, frameon=False, facecolor="1",figsize=(15,15))
        for ii in iiv:
            one_lc_iapp=[]
            if abs(i_ap-INFO['node_pars'+str(ii)]['I_app'])<0.1:
                saddle_point=INFO['Jac_info'+str(ii)][0]
                m_Jmat=INFO['Jac_info'+str(ii)][1]
                Jeival=INFO['Jac_info'+str(ii)][2]
                Jeivect=INFO['Jac_info'+str(ii)][3]
                try:
                    c_lc_iap=0
                    for ilc_iap in INFO['One_lc_iapp']:
                        if abs(i_ap-ilc_iap)<0.1:
                            one_lc_iapp=c_lc_iap
                        c_lc_iap+=1
                except:
                    pass
                if one_lc_iapp !=[]:
                    one_lc=INFO['One_lc'][one_lc_iapp]
                    one_lc_s=INFO['One_lc_s'][one_lc_iapp]
                    for jj in v_coords:
                        if max(one_lc[v_coords[0]])<=1:
                            dxx=scale_manifold*1.0
                            ax.set_xlim(-0.1,1.1)
                            if axi!=[]:
                                axi.set_xlim(-0.1,1.1)
                        if max(one_lc[v_coords[0]])>1:
                            dxx=scale_manifold*180.0
                            ax.set_xlim(-100,80)
                            if axi!=[]:
                                axi.set_xlim([-100,80])

                        if max(one_lc[v_coords[1]])<=1:
                            dyy=scale_manifold*1.0
                            ax.set_ylim(-0.1,1.1)
                            if axi!=[]:
                                axi.set_ylim(-0.1,1.1)
                        if max(one_lc[v_coords[1]])>1:
                            dyy=scale_manifold*180.0
                            ax.set_ylim(-100,80)
                            if axi!=[]:
                                axi.set_ylim(-100,80)

                    c=0
                else:
                    one_lc=[]
                    one_lc_s=[]
                    if max(abs(saddle_point[v_coords[0]]))<=1:
                        dxx=scale_manifold*1.0
                        ax.set_xlim(-0.1,1.1)
                        if axi!=[]:
                            axi.set_xlim(-0.1,1.1)
                    else:
                        dxx=scale_manifold*180.0
                        ax.set_xlim(-100,80)
                        if axi!=[]:
                            axi.set_xlim([-100,80])
                    if max(abs(saddle_point[v_coords[1]]))<=1:
                        dyy=scale_manifold*1.0
                        ax.set_ylim(-0.1,1.1)
                        if axi!=[]:
                            axi.set_ylim(-0.1,1.1)
                    else:
                        dyy=scale_manifold*180.0
                        ax.set_ylim(-100,80)
                        if axi!=[]:
                            axi.set_ylim([-100,80])


                    if any(Jeival.real>0):
                        ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=30,mfc='none')
                    else:
                        ax.plot(saddle_point[i],saddle_point[j],saddle_point[k],'o',markersize=30)
                    # ax.quiver(X,Y,Z,W,A,B, length=0.1)
                    try:
                        ax.set_xlabel('$'+one_lc_s[i]+'$',fontsize=size_axis_font)
                        ax.set_ylabel('$'+one_lc_s[j]+'$',fontsize=size_axis_font)
                        ax.set_zlabel('$'+one_lc_s[k]+'$',fontsize=size_axis_font)
                    except:
                        pass
            ###############################################3
                if small_marker:
                    marker=14
                else:
                    marker=60
                if k==[]:
                    if one_lc!=[]:
                        ax.plot(one_lc[i],one_lc[j],color=[0,0,0],linewidth=4.0)
                        if axi!=[] and abs(i_api-i_ap)<0.1:
                            axi.plot(one_lc[i],one_lc[j],color=[0,0,0],linewidth=4.0)
                    if any(Jeival.real>0):
                        ax.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=marker)
                        ax.plot(saddle_point[i],saddle_point[j],'o',color=[1,1,1],markersize=int(marker/2))
                        if axi!=[] and abs(i_api-i_ap)<0.1:
                            axi.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=marker)
                            axi.plot(saddle_point[i],saddle_point[j],'o',color=[1,1,1],markersize=int(marker/2))
                    else:
                        ax.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=marker)
                        if axi!=[] and abs(i_api-i_ap)<0.1:
                            axi.plot(saddle_point[i],saddle_point[j],'o',color=[0,0,0],markersize=marker)
                        str_snpos=[saddle_point[i],saddle_point[j]]

                    v_coords=[i,j]
                    if plt_Jvect:
                        # for i_j in range(len(saddle_point)):
                        mJeival=mean(abs(Jeival.real))
                        n_arrows=0
                        n_arrows0=-0.5
                        for i_j in np.argsort(abs(Jeival)):
                        # for i_j in range(len(saddle_point)):
                            n_arrows0+=0.5
                            n_arrows=1+int(n_arrows0)
                            y1 = [saddle_point[i]]
                            y2 = [saddle_point[j]]
                            Y1, Y2 = np.meshgrid(y1, y2)
                            u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
                            NI, NJ = Y1.shape
                            for ii_i in range(NI):
                                for jj_j in range(NJ):
                                    x = Y1[ii_i, jj_j]
                                    y = Y2[ii_i, jj_j]
                                    u[ii_i, jj_j] = dxx*Jeivect[i_j][v_coords[0]].real/norm(Jeivect[i_j].real)
                                    v[ii_i, jj_j] = dyy*Jeivect[i_j][v_coords[1]].real/norm(Jeivect[i_j].real)
                                    if vis_easy:
                                        u[ii_i, jj_j] = dxx*Jeivect[i_j][v_coords[0]]/norm(Jeivect[i_j][v_coords].real)
                                        v[ii_i, jj_j] = dyy*Jeivect[i_j][v_coords[1]]/norm(Jeivect[i_j][v_coords].real)
                            # print('Stable '+ str(u**2+v**2))
                            if Jeival[i_j].real<0:
                                for cc_arrows in range(n_arrows):
                                    if n_arrows==1:
                                        ax.annotate('', (saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))), (u+saddle_point[v_coords[0]], v+saddle_point[v_coords[1]]),
                                            ha="right", va="center",
                                            size=size_axis_font,
                                            arrowprops=dict(arrowstyle='->',
                                                            # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                            lw=2.0,
                                                            # patchB=p,
                                                            shrinkA=5,
                                                            shrinkB=5,
                                                            fc="k", ec="k",
                                                            connectionstyle="arc3,rad=-0.0",
                                                            ))
                                        ax.annotate('', (saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))), (-u+saddle_point[v_coords[0]], -v+saddle_point[v_coords[1]]),
                                            ha="right", va="center",
                                            size=size_axis_font,
                                            arrowprops=dict(arrowstyle='->',
                                                            # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                            lw=2.0,
                                                            # patchB=p,
                                                            shrinkA=5,
                                                            shrinkB=5,
                                                            fc="k", ec="k",
                                                            connectionstyle="arc3,rad=0.0",
                                                            ))
                                        if axi!=[] and abs(i_api-i_ap)<0.1:
                                            axi.annotate('', (saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))), (u+saddle_point[v_coords[0]], v+saddle_point[v_coords[1]]),
                                                ha="right", va="center",
                                                size=size_axis_font,
                                                arrowprops=dict(arrowstyle='->',
                                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                                lw=2.0,
                                                                # patchB=p,
                                                                shrinkA=5,
                                                                shrinkB=5,
                                                                fc="k", ec="k",
                                                                connectionstyle="arc3,rad=-0.0",
                                                                ))
                                            axi.annotate('', (saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))), (-u+saddle_point[v_coords[0]], -v+saddle_point[v_coords[1]]),
                                                ha="right", va="center",
                                                size=size_axis_font,
                                                arrowprops=dict(arrowstyle='->',
                                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                                lw=2.0,
                                                                # patchB=p,
                                                                shrinkA=5,
                                                                shrinkB=5,
                                                                fc="k", ec="k",
                                                                connectionstyle="arc3,rad=0.0",
                                                                ))
                            if Jeival[i_j].real>0:
                                if n_arrows==6:
                                    for cc_arrows in range(n_arrows):
                                        ax.annotate('', (u+saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), v+saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                            #xycoords="figure fraction", textcoords="figure fraction",
                                            ha="right", va="center",
                                            size=size_axis_font,
                                            arrowprops=dict(arrowstyle='->',
                                                            # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                            lw=2.0,
                                                            # patchB=p,
                                                            shrinkA=5,
                                                            shrinkB=5,
                                                            fc="k", ec="k",
                                                            connectionstyle="arc3,rad=-0.0",
                                                            ))
                                        ax.annotate('',(-u+saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), -v+saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                            #xycoords="figure fraction", textcoords="figure fraction",
                                            ha="right", va="center",
                                            size=size_axis_font,
                                            arrowprops=dict(arrowstyle='->',
                                                            # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                            lw=2.0,
                                                            # patchB=p,
                                                            shrinkA=5,
                                                            shrinkB=5,
                                                            fc="k", ec="k",
                                                            connectionstyle="arc3,rad=-0.0",
                                                            ))
                                        if axi!=[] and abs(i_api-i_ap)<0.1:
                                            axi.annotate('', (u+saddle_point[v_coords[0]]-u*(cc_arrows/float(n_arrows)), v+saddle_point[v_coords[1]]-v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                                #xycoords="figure fraction", textcoords="figure fraction",
                                                ha="right", va="center",
                                                size=size_axis_font,
                                                arrowprops=dict(arrowstyle='->',
                                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                                lw=2.0,
                                                                # patchB=p,
                                                                shrinkA=5,
                                                                shrinkB=5,
                                                                fc="k", ec="k",
                                                                connectionstyle="arc3,rad=-0.0",
                                                                ))
                                            axi.annotate('',(-u+saddle_point[v_coords[0]]+u*(cc_arrows/float(n_arrows)), -v+saddle_point[v_coords[1]]+v*(cc_arrows/float(n_arrows))),(saddle_point[v_coords[0]], saddle_point[v_coords[1]]),
                                                #xycoords="figure fraction", textcoords="figure fraction",
                                                ha="right", va="center",
                                                size=size_axis_font,
                                                arrowprops=dict(arrowstyle='->',
                                                                # lw=exp(norm(Jeival[i_j].real)/norm(Jeival))*5.0,
                                                                lw=2.0,
                                                                # patchB=p,
                                                                shrinkA=5,
                                                                shrinkB=5,
                                                                fc="k", ec="k",
                                                                connectionstyle="arc3,rad=-0.0",
                                                                ))

        # ax.set_xlim([-0.1,1.1])
        # ax.set_ylim([-500,80])
        ax.axis('off')
        if axi!=[] and abs(i_api-i_ap)<0.1:
            # axi.set_xlim([-0.1,1.1])
            # axi.set_ylim([-500,80])
            # axi.axis('off')
            pass
        if Ei!=[]:
            for eii in Ei:
                ax.plot(np.linspace(-0.1,1.1,50), np.squeeze(eii*np.ones(50)),linewidth=6,color=[0.9,0.9,0.9])
                if axi!=[] and abs(i_api-i_ap)<0.1:
                    axi.plot(np.linspace(-0.1,1.1,50),np.squeeze(eii*np.ones(50)),linewidth=6,color=[0.9,0.9,0.9])
        if dir_figs!=[]:
            ax.set_xlim([-0.1,1.1])
            ax.set_ylim([-500,80])
            ax.axis('off')
            if axi!=[] and abs(i_api-i_ap)<0.1:
                axi.set_xlim([-0.1,1.1])
                axi.set_ylim([-500,80])
                axi.axis('off')
                pass
            name_file='Orbit_snap_in_Bif_Diagram_Nai_'+str(int(INFO['node_pars'+str(ii)]['Na_i']))+'_Ki_'+str(int(INFO['node_pars'+str(ii)]['K_i']))+'_Ko_'+str(int(INFO['node_pars'+str(ii)]['K_o']))+'_Iapp_'+str(int(i_ap))
            md_dir={
             'Title':'Frozen Orbit graph',
             'Author':"Susana Contreras 24/01/2018",
             'Subject':'Bifuration Diagram Nai='+str(INFO['node_pars'+str(ii)]['Na_i'])+' Ki='+str(INFO['node_pars'+str(ii)]['K_i'])+' Iapp='+str(i_ap),
             'Keywords':"Na-K-Pump, dependent excitability, adaptation, Bif analysis"
             }
            fig_file_title=name_file
            saving_pdf_figure(fig,dir_figs+fig_file_title,md_dir,pickle_f=False)
            v_figs.append(dir_figs+fig_file_title)

        if dir_figs==[]:
            # v_figs.append(copy(ax))
            if i_api!=[] and axi!=[]:
                v_figs=axi
                plt.close("fig")
        v_iapp.append(i_ap)
        v_ina.append(INFO['node_pars'+str(ii)]['Na_i'])
        v_iki.append(INFO['node_pars'+str(ii)]['K_i'])
        v_iko.append(INFO['node_pars'+str(ii)]['K_o'])
    return v_figs,v_iapp,v_ina,v_iko,v_iki
