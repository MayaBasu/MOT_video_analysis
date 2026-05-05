import numpy as np
import matplotlib.pyplot as plt

gamma_D2_85 = 2* 6.0666*np.pi #lifetime of the D2 transition of rubidium 85 in MHz
def scatter_rate(delta,Gamma,s):
    return (Gamma/2)*(s/(1 + s + (2*delta/Gamma)**2))

def doppler(delta,k,v):
    return delta - k*v

def doppler_force(detune,k,x,s,gamma):
    return scatter_rate(doppler(detune,k,x),gamma,s)-scatter_rate(doppler(detune, -k, x), gamma, s)


#def cooled_velocity(v0,dt,steps):
#   velocities= [v0]
#    times = [0]
#    for step in range(steps):
#        times.append(times[-1]+dt)
#        velocities.append(velocities[-1] + doppler_force(times[-1],v0,dt,step))

def plot_scatter_rate():
    s = 1 # assume fully saturated
    example_detune = -10
    x = np.linspace(-30 ,30 ,100)
    fig, ax = plt.subplots(nrows = 2 ,ncols = 1,figsize = (10,10))

    ax[0].plot(x,scatter_rate(x,gamma_D2_85,s), label="Rubidium 85 D2 scatter rate")
    ax[0].scatter(example_detune, scatter_rate(example_detune, gamma_D2_85, s),label="Example Detuning")
    ax[0].set_xlabel("Detuning $\omega_{laser} - \omega_0$[MHz]")
    ax[0].set_ylabel("Scatter Rate [MHz]")
    ax[0].set_title(f"Rubidium 85 D2 Scatter Rate vs Laser Frequency Detuning for $s = {s}$")
    ax[0].legend()
    x = np.linspace(-1e-5, 1e-5, 100)
    ax[1].plot(x, doppler_force(example_detune,1,x,s,gamma_D2_85),label="Radiation pressure on Rb 85 in detuned laser light")

   # ax[1].plot(x,scatter_rate(doppler(example_detune, -1, x), 1, s))
    ax[1].set_xlabel("Velocity [$k_{laser}$]")
    ax[1].set_ylabel("Force [$\hbar k_{laser}$]")
    ax[1].set_title(f"Radiation pressure on a Rubidium 85 atom as a function of velocity inside detuned light")
    ax[1].legend()

    plt.suptitle("Doppler Cooling")
    plt.tight_layout()
    plt.savefig("poster_plots/doppler_cooling.png")

    plt.show()

plot_scatter_rate()
