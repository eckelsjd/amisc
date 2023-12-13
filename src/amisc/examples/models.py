"""Providing some example test functions"""
from pathlib import Path
import pickle
import uuid

import numpy as np

from amisc.system import ComponentSpec, SystemSurrogate
from amisc.rv import UniformRV, NormalRV


def tanh_func(x, *args, A=2, L=1, frac=4, **kwargs):
    """Simple tunable tanh function"""
    return {'y': A*np.tanh(2/(L/frac)*(x-L/2)) + A + 1}


def ishigami(x, *args, a=7.0, b=0.1, **kwargs):
    """For testing Sobol indices: https://doi.org/10.1109/ISUMA.1990.151285"""
    return {'y': np.sin(x[..., 0:1]) + a*np.sin(x[..., 1:2])**2 + b*(x[..., 2:3]**4)*np.sin(x[..., 0:1])}


def borehole_func(x, *args, **kwargs):
    """Model found at https://www.sfu.ca/~ssurjano/borehole.html
    :returns vdot: Water flow rate in m^3/yr
    """
    rw = x[..., 0]      # Radius of borehole (m)
    r = x[..., 1]       # Radius of influence (m)
    Tu = x[..., 2]      # Transmissivity (m^2/yr)
    Hu = x[..., 3]      # Potentiometric head (m)
    Tl = x[..., 4]      # Transmissivity (m^2/yr)
    Hl = x[..., 5]      # Potentiometric head (m)
    L = x[..., 6]       # Length of borehole (m)
    Kw = x[..., 7]      # Hydraulic conductivity (m/yr)
    vdot = 2*np.pi*Tu*(Hu-Hl) / (np.log(r/rw) * (1 + (2*L*Tu/(np.log(r/rw)*Kw*rw**2)) + (Tu/Tl)))

    return {'y': vdot[..., np.newaxis]}


def wing_weight_func(x, *args, **kwargs):
    """Model found at https://www.sfu.ca/~ssurjano/wingweight.html
    :returns Wwing: the weight of the airplane wing (lb)
    """
    Sw = x[..., 0]      # Wing area (ft^2)
    Wfw = x[..., 1]     # Weight of fuel (lb)
    A = x[..., 2]       # Aspect ratio
    Lambda = x[..., 3]  # Quarter-chord sweep (deg)
    q = x[..., 4]       # Dynamic pressure (lb/ft^2)
    lamb = x[..., 5]    # taper ratio
    tc = x[..., 6]      # Aerofoil thickness to chord ratio
    Nz = x[..., 7]      # Ultimate load factor
    Wdg = x[..., 8]     # Design gross weight (lb)
    Wp = x[..., 9]      # Paint weight (lb/ft^2)
    Lambda = Lambda*(np.pi/180)
    Wwing = 0.036*(Sw**0.758)*(Wfw**0.0035)*((A/(np.cos(Lambda))**2)**0.6)*(q**0.006)*(lamb**0.04)*\
            (100*tc/np.cos(Lambda))**(-0.3)*((Nz*Wdg)**0.49) + Sw*Wp

    return {'y': Wwing[..., np.newaxis]}


def nonlinear_wave(x, *args, env_var=0.1**2, wavelength=0.5, wave_amp=0.1, tanh_amp=0.5, L=1, t=0.25, **kwargs):
    """Custom nonlinear model of a traveling Gaussian wave for testing.

    :param x: `(..., x_dim)`, input locations
    :param env_var: variance of Gaussian envelope
    :param wavelength: sinusoidal perturbation wavelength
    :param wave_amp: amplitude of perturbation
    :param tanh_amp: amplitude of tanh(x)
    :param L: domain length of underlying tanh function
    :param t: transition length of tanh function (as fraction of L)
    :returns: `(..., y_dim)`, model output
    """
    # Traveling sinusoid with moving Gaussian envelope (theta is x2)
    env_range = [0.2, 0.6]
    mu = env_range[0] + x[..., 1] * (env_range[1] - env_range[0])
    theta_env = 1 / (np.sqrt(2 * np.pi * env_var)) * np.exp(-0.5 * (x[..., 0] - mu) ** 2 / env_var)
    ftheta = wave_amp * np.sin((2*np.pi/wavelength) * x[..., 1]) * theta_env

    # Underlying tanh dependence on x1
    fd = tanh_amp * np.tanh(2/(L*t)*(x[..., 0] - L/2)) + tanh_amp

    # Compute model = f(theta, d) + f(d)
    y = np.expand_dims(ftheta + fd, axis=-1)  # (..., 1)

    return {'y': y}


def fire_sat_system(save_dir=None):
    """Fire satellite detection system model from Chaudhuri 2018.

    !!! Note "Some modifications"
        Orbit will save outputs all the time; Power won't because it is part of an FPI loop. Orbit and Power can
        return `np.nan` some of the time (optional, to test robustness of surrogate). Power and Attitude have an `alpha` fidelity
        index that controls accuracy of the returned values. `alpha=0,1,2` corresponds to `8,4,0` percent error.

    :param save_dir: where to save model outputs
    :returns: a `SystemSurrogate` object for the fire sat MD example system
    """
    Re = 6378140    # Radius of Earth (m)
    mu = 3.986e14   # Gravitational parameter (m^3 s^-2)
    eta = 0.22      # Power efficiency
    Id = 0.77       # Inherent degradation of the array
    thetai = 0      # Sun incidence angle
    LT = 15         # Spacecraft lifetime (years)
    eps = 0.0375    # Power production degradation (%/year)
    rlw = 3         # Length to width ratio
    nsa = 3         # Number of solar arrays
    rho = 700       # Mass density of arrays (kg/m^3)
    t = 0.005       # Thickness (m)
    D = 2           # Distance between panels (m)
    IbodyX = 6200   # kg*m^2
    IbodyY = 6200   # kg*m^2
    IbodyZ = 4700   # kg*m^2
    dt_slew = 760   # s
    theta = 15      # Deviation of moment axis from vertical (deg)
    As = 13.85      # Area reflecting radiation (m^2)
    c = 2.9979e8    # Speed of light (m/s)
    M = 7.96e15     # Magnetic moment of earth (A*m^2)
    Rd = 5          # Residual dipole of spacecraft (A*m^2)
    rhoa=5.148e-11  # Atmospheric density (kg/m^3) -- typo in Chaudhuri 2018 has this as 1e11 instead
    A = 13.85       # Cross-section in flight (m^2)
    Phold = 20      # Holding power (W)
    omega = 6000    # Max vel of wheel (rpm)
    nrw = 3         # Number of reaction wheels

    def orbit_fun(x, output_dir=None, pct_failure=0):
        H = x[..., 0:1]                         # Altitude (m)
        phi = x[..., 1:2]                       # Target diameter (m)
        vel = np.sqrt(mu / (Re + H))            # Satellite velocity (m/s)
        dt_orbit = 2*np.pi*(Re + H) / vel       # Orbit period (s)
        dt_eclipse = (dt_orbit/np.pi)*np.arcsin(Re / (Re + H))  # Eclipse period (s)
        theta_slew = np.arctan(np.sin(phi / Re) / (1 - np.cos(phi / Re) + H/Re))    # Max slew angle (rad)
        if np.random.rand() < pct_failure:
            i = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            i2 = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            vel[i + (0,)] = np.nan
            theta_slew[i2 + (0,)] = np.nan
        y = np.concatenate((vel, dt_orbit, dt_eclipse, theta_slew), axis=-1)
        if output_dir is not None:
            files = []
            id = str(uuid.uuid4())
            for index in np.ndindex(*x.shape[:-1]):
                fname = f'{id}_{index}.pkl'
                with open(Path(output_dir) / fname, 'wb') as fd:
                    pickle.dump({'y': y[index + (slice(None),)]}, fd)
                files.append(fname)
            return {'y': y, 'files': files}
        else:
            return {'y': y}

    def power_fun(x, alpha=(0,), *, output_dir=None, pct_failure=0):
        pct = 1 - (2 - alpha[0]) * 0.04 if len(alpha) == 1 else 1  # extra pct error term
        Po = x[..., 0:1]            # Other power sources (W)
        Fs = x[..., 1:2]            # Solar flux (W/m^2)
        dt_orbit = x[..., 2:3]      # Orbit period (s)
        dt_eclipse = x[..., 3:4]    # Eclipse period (s)
        Pacs = x[..., 4:5]          # Power from attitude control system (W)
        Ptot = Po + Pacs
        Pe = Ptot
        Pd = Ptot
        Xe = 0.6                     # These are power efficiencies in eclipse and daylight
        Xd = 0.8                     # See Ch. 11 of Wertz 1999 SMAD
        Te = dt_eclipse
        Td = dt_orbit - Te
        Psa = ((Pe*Te/Xe) + (Pd*Td/Xd)) / Td
        Pbol = eta*Fs*Id*np.cos(thetai)
        Peol = Pbol * (1 - eps)**LT
        Asa = Psa / Peol            # Total solar array size (m^2)
        L = np.sqrt(Asa*rlw/nsa)
        W = np.sqrt(Asa/(rlw*nsa))
        msa = 2*rho*L*W*t           # Mass of solar array
        Ix = msa*((1/12)*(L**2 + t**2) + (D+L/2)**2) + IbodyX
        Iy = (msa/12)*(L**2 + W**2) + IbodyY  # typo in Zaman 2013 has this as H**2 instead of L**2
        Iz = msa*((1/12)*(L**2 + W**2) + (D + L/2)**2) + IbodyZ
        Itot = np.concatenate((Ix, Iy, Iz), axis=-1)
        Imin = np.min(Itot, axis=-1, keepdims=True)
        Imax = np.max(Itot, axis=-1, keepdims=True)
        if np.random.rand() < pct_failure:
            i = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            i2 = tuple([np.random.randint(0, N) for N in x.shape[:-1]])
            Imin[i2 + (0,)] = np.nan
            Asa[i + (0,)] = np.nan
        y = np.concatenate((Imin, Imax*pct, Ptot*pct, Asa), axis=-1)

        if output_dir is not None:
            files = []
            id = str(uuid.uuid4())
            for index in np.ndindex(*x.shape[:-1]):
                fname = f'{id}_{index}.pkl'
                with open(Path(output_dir) / fname, 'wb') as fd:
                    pickle.dump({'y': y[index + (slice(None),)]}, fd)
                files.append(fname)
            return {'y': y, 'files': files}
        else:
            return {'y': y}

    def attitude_fun(x, /, alpha=(0,)):
        pct = 1 - (2 - alpha[0])*0.04 if len(alpha) == 1 else 1  # extra model fidelity pct error term
        H = x[..., 0:1]             # Altitude (m)
        Fs = x[..., 1:2]            # Solar flux
        Lsp = x[..., 2:3]           # Moment arm for solar radiation pressure
        q = x[..., 3:4]             # Reflectance factor
        La = x[..., 4:5]            # Moment arm for aerodynamic drag
        Cd = x[..., 5:6]            # Drag coefficient
        vel = x[..., 6:7]           # Satellite velocity
        theta_slew = x[..., 7:8]    # Max slew angle
        Imin = x[..., 8:9]          # Minimum moment of inertia
        Imax = x[..., 9:10]         # Maximum moment of inertia
        tau_slew = 4*theta_slew*Imax / dt_slew**2
        tau_g = 3*mu*np.abs(Imax - Imin)*np.sin(2*theta*(np.pi/180)) / (2*(Re+H)**3)
        tau_sp = Lsp*Fs*As*(1+q)*np.cos(thetai) / c
        tau_m = 2*M*Rd / (Re + H)**3
        tau_a = (1/2)*La*rhoa*Cd*A*vel**2
        tau_dist = np.sqrt(tau_g**2 + tau_sp**2 + tau_m**2 + tau_a**2)
        tau_tot = np.max(np.concatenate((tau_slew, tau_dist), axis=-1), axis=-1, keepdims=True)
        Pacs = tau_tot*(omega*(2*np.pi/60)) + nrw*Phold
        y = np.concatenate((Pacs, tau_tot), axis=-1) * pct
        return {'y': y}

    orbit = ComponentSpec(orbit_fun, name='Orbit', exo_in=[0, 1], coupling_out=[0, 1, 2, 3], max_beta=(3, 3),
                          model_kwargs={'pct_failure': 0}, save_output=True)
    power = ComponentSpec(power_fun, name='Power', truth_alpha=(2,), exo_in=[2, 3], max_alpha=(2,), max_beta=(3,)*5,
                          coupling_in={'Orbit': [1, 2], 'Attitude': [0]}, coupling_out=[4, 5, 6, 7], save_output=True,
                          model_kwargs={'pct_failure': 0})
    attitude = ComponentSpec(attitude_fun, name='Attitude', truth_alpha=2, max_alpha=2, max_beta=(3,)*10,
                             exo_in=[0, 3, 4, 5, 6, 7], coupling_in={'Orbit': [0, 3], 'Power': [0, 1]},
                             coupling_out=[8, 9])

    exo_vars = [NormalRV(18e6, 1e6, 'H'), NormalRV(235e3, 10e3, '\u03D5'), NormalRV(1000, 50, 'Po'),
                NormalRV(1400, 20, 'Fs'), NormalRV(2, 0.4, 'Lsp'), NormalRV(0.5, 0.1, 'q'),
                NormalRV(2, 0.4, 'La'), NormalRV(1, 0.2, 'Cd')]
    coupling_vars = [UniformRV(2000, 6000, 'Vsat'), UniformRV(20000, 60000, 'To'), UniformRV(1000, 5000, 'Te'),
                     UniformRV(0, 4, 'Slew'), UniformRV(0, 12000, 'Imin'), UniformRV(0, 12000, 'Imax'),
                     UniformRV(0, 10000, 'Ptot'), UniformRV(0, 50, 'Asa'), UniformRV(0, 100, 'Pat'),
                     UniformRV(0, 5, 'tau_tot')]
    surr = SystemSurrogate([orbit, power, attitude], exo_vars, coupling_vars, est_bds=500, save_dir=save_dir)

    return surr
