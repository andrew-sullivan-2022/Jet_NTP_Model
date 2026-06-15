import numpy as np
import matplotlib
from pydl.pydlutils.cooling import read_ds_cooling

## AGN HEATING MODEL

## Physical constants
# speed of light
c_light = 3*10**8 # m s^-1
# newton's gravitational constant
G = 6.6743*10**(-11) # m^3 kg^-1 s^-2
# the bolztmann constant
kB = 1.3806*10**(-23) # m^2 kg s^-2 K^-1
# proton mass
mp = 1.6726*10**(-27) # kg
# electron volt
eV = 1.602*10**(-19) # J

## Cosmological constants
# hubble parameter
h = 0.6751
# present-day critical density of the universe
rho_crit = 1.8788*10**(-26)*h**2 # kg m^-3
# density parameters
Omega_b = 0.0486
Omega_DM = 0.2589
# fraction of baryon content
f_b = Omega_b/(Omega_b + Omega_DM)

## Unit conversions
# solar mass
Msol = 1.989*10**30 # kg
# year
yr = 3.156*10**7 # s
# megayear
Myr = 3.156*10**13 # s
# gigayear
Gyr = 3.156*10**16 # s
# kiloparsec
kpc = 3.086*10**19 # m
# megaparsec
Mpc = 3.086*10**22 # m

## Gas parameters
# mean molecular weight
mu = 0.6
# mean electron weight
mu_e = 1.15
# mean ion weight
mu_i = (1/mu + 1/mu_e)**(-1)
# adiabatic index
gamma = 5/3

## Functions
# saving files (as nested arrays)
def save_nested_arrays(filename, data):
    ## data shape: [N][M][X]
    with open(filename, "w") as f:
        for i, block in enumerate(data):      # N blocks
            for array in block:               # M arrays
                np.savetxt(f, np.atleast_1d(array)[None, :])
            if i != len(data) - 1:
                f.write("\n")

## Jet ensemble-average heating model
# function for calculating heating rate, velocity kick and NTP fraction
def calculate_ensemble_jet_heating(log_Qjet_vals, log_age_vals, duty_cycle, gas_density_profile, temperature_profile, halo_radius):
    ## Inputs
    # loading jet powers
    log_Qjet_vals = np.atleast_1d(log_Qjet_vals) # log W
    Qjet_vals = np.power(10, log_Qjet_vals) # W
    power_res = 1 if np.isscalar(log_Qjet_vals) else len(log_Qjet_vals)
    # loading active ages
    log_age_vals = np.atleast_1d(log_age_vals) # log yr
    age_vals = np.power(10, log_age_vals)*yr # s
    age_res = 1 if np.isscalar(log_age_vals) else len(log_age_vals)
    # sample size
    if power_res != age_res:
        print('Sampled jet powers and ages do not match in size')
    sample_size = power_res
    # loading duty cycles
    duty_cycle = np.atleast_1d(duty_cycle) # %
    duty_cycle_res = 1 if np.isscalar(duty_cycle) else len(duty_cycle)
    # halo radius resolution
    radius_bins = len(halo_radius)

    ## AGN jet parameters
    # jet beam power constant
    k1 = 1
    # normalisation radius for jet environment
    normalisation_index = np.argmin(np.abs(halo_radius - 10*kpc))
    # jet half-opening angle
    thetaJ = 8.5*np.pi/180  # rads
    # FR-I half-opening angle
    thetaFRI = 15*np.pi/180  # rads
    # jet solid angle
    Omega = 2*np.pi*(1 - np.cos(thetaJ)) # sr

    ## Environmental profiles
    # number density profiles
    number_density_profile = np.divide(gas_density_profile, (mu*mp))  # m^-3
    hydrogen_density_profile = np.multiply((4/9), number_density_profile)  # m^-3
    # gas density negative inner log slope
    gas_density_derivative = np.gradient(gas_density_profile, halo_radius) # kg m^-4
    gas_density_log_slope = np.multiply(gas_density_derivative, np.divide(halo_radius, gas_density_profile))
    e = -np.average(gas_density_log_slope[0:np.argmin(np.abs(halo_radius - 20 * kpc))])
    # thermal pressure profiles
    th_pressure_profile = np.multiply(gas_density_profile, kB*temperature_profile /(mu*mp))  # Pa
    th_pressure_derivative = np.gradient(th_pressure_profile, halo_radius)  # Pa m^-1
    # sound speed, thermal velocity
    sound_speed_profile = np.sqrt(np.multiply(temperature_profile, (gamma*kB/(mu*mp))))  # m s^-1
    th_velocity_profile = np.sqrt(np.multiply(temperature_profile, 3*kB/(mu*mp)))  # m s^-1

    ## Cooling function
    # loading cooling function
    abundance_file = 'm-05.cie'  # metallicity for Sutherland & Dopita (1993) cooling function, here corresponding to Z = 0.3 Zsolar
    # interpolating from the gas temperature to the cooling function
    logT_cool, logLambda_cool = read_ds_cooling(abundance_file)  # T in K, Lambda 13 dex higher than SI value
    logLambda_cool_interp = np.interp(np.log10(temperature_profile), logT_cool, logLambda_cool - 13)  # log ( W m^3 )
    # interpolated cooling function
    Lambda_cool_interp = np.asarray([10**i for i in logLambda_cool_interp])  # W m^3
    # cooling rate
    cooling_rate = np.multiply(np.square(hydrogen_density_profile), Lambda_cool_interp)  # W m^-3

    ## Escaping the brightest cluster galaxy (BCG)
    # time offset as a function of jet power, calibrated from Young+2025 in prep
    def t_min(Qjet_vals, e):  # s
        return 1.27*Myr*(4/(4 - e))*((3 - e)/3)**0.5*(10**37/Qjet_vals)**0.5

    ## Removing jets that cannot escape the central galaxy
    # jet outburst time and escape time
    t_outburst_vals, t_esc_vals, stuck_jet_indices = np.empty(sample_size), np.empty(sample_size), np.empty(sample_size)
    for i in range(sample_size):
        if age_vals[i] > t_min(Qjet_vals[i], e):
            stuck_jet_indices[i] = np.nan
            t_esc_vals[i] = t_min(Qjet_vals[i], e)  # s
            t_outburst_vals[i] = age_vals[i] - t_min(Qjet_vals[i], e)  # s
        elif age_vals[i] <= t_min(Qjet_vals[i], e):
            stuck_jet_indices[i] = i
            t_esc_vals[i] = np.nan
            t_outburst_vals[i] = np.nan
    stuck_jet_indices = stuck_jet_indices[~np.isnan(stuck_jet_indices)]
    stuck_jet_indices = [int(i) for i in stuck_jet_indices]
    # updating jet powers and jet times to remove jets that do not escape
    Qjet_vals = [i for j, i in enumerate(Qjet_vals) if j not in stuck_jet_indices] # W
    t_outburst_vals = [i for j, i in enumerate(t_outburst_vals) if j not in stuck_jet_indices] # s
    # number and fraction of jets that escape the central galaxy
    N_esc = len(t_outburst_vals)

    ## Removing jets that never reach pressure equilibrium
    # jet equilibrium time
    t_eq_vals = (2/(4 - e))*(k1/(Omega * c_light))**((2 - e)/4)*(gas_density_profile[normalisation_index]*(halo_radius[normalisation_index])**e)**0.5/((th_pressure_profile[normalisation_index])**((4 - e)/4))*np.power(Qjet_vals, ((2 - e)/4))  # s
    does_not_flare_indices = np.empty(N_esc)
    for i in range(N_esc):
        if t_outburst_vals[i] > t_eq_vals[i]:
            does_not_flare_indices[i] = np.nan
        elif t_outburst_vals[i] <= t_eq_vals[i]:
            does_not_flare_indices[i] = i
    does_not_flare_indices = does_not_flare_indices[~np.isnan(does_not_flare_indices)]
    does_not_flare_indices = [int(i) for i in does_not_flare_indices]
    # updating jet powers and jet times to remove non-flaring jets
    Qjet_vals = [i for j, i in enumerate(Qjet_vals) if j not in does_not_flare_indices] # W
    t_outburst_vals = [i for j, i in enumerate(t_outburst_vals) if j not in does_not_flare_indices] # s
    t_eq_vals = [i for j, i in enumerate(t_eq_vals) if j not in does_not_flare_indices] # s
    # number and fraction of jets that reach pressure equilibrium
    N_flare = len(t_outburst_vals)

    ## Jet evolution model
    # equilibrium jet lengths
    R_eq_vals = (k1/(Omega*c_light*th_pressure_profile[normalisation_index]))**0.5*np.sqrt(Qjet_vals)  # m
    # jet length offset from 'flaring jet phase' up to pressure equilibrium
    x_eq_vals = np.tan(thetaJ)/np.tan(thetaFRI)*R_eq_vals  # m
    # 'flaring jet' time offset
    t_fj_vals = (1/2)*((4 - e)/(3 - e))*np.multiply(np.divide(x_eq_vals, R_eq_vals), t_eq_vals)  # s
    # jet outburst length
    adjusted_time_samples = np.add(np.subtract(t_outburst_vals, t_eq_vals), t_fj_vals)  # s
    R_outbust_vals = np.add(np.subtract(R_eq_vals, x_eq_vals), np.multiply(x_eq_vals, np.power(np.divide(adjusted_time_samples, t_fj_vals), 1/(3 - e))))  # m
    # jet outburst volume
    volume_vals = np.empty(N_flare)
    for i in range(N_flare):
        R = np.logspace(np.log10(R_eq_vals[i]), np.log10(R_outbust_vals[i]), base=10, num=200)
        volume_vals[i] = 4*np.pi*np.trapz(R**2*(1 - R/np.sqrt(R**2 + (np.tan(thetaFRI)*(R - R_eq_vals[i] + x_eq_vals[i]))**2)), R) # m^3
    volume_vals = np.add(4*np.pi*(1 - np.cos(thetaJ))*np.power(R_eq_vals, 3)/3, volume_vals)  # m^3

    ## Calculating the velocity kicks in the jet heating model
    # gas velocity kicks per energy injection rate
    velocity_per_volumetric_energy_rate = np.divide((gamma - 1), (np.add(th_pressure_derivative, np.multiply(2*gamma, np.divide(th_pressure_profile, halo_radius)))))  # m^2 s^2 kg^-1
    # the volumetric heating rate
    heating_rate_vals = np.divide(np.multiply(Qjet_vals, 2), volume_vals)  # W m^-3
    # the heating-induced velocity kick
    heating_v_kick_vals = np.empty((N_flare, radius_bins))
    for i in range(N_flare):
        heating_v_kick_vals[i] = velocity_per_volumetric_energy_rate*heating_rate_vals[i]  # m s^-1
    # the cooling-induced velocity kick
    cooling_v_kick = np.multiply(cooling_rate, velocity_per_volumetric_energy_rate)  # m s^-1

    ## Calculating the effective feedback profiles
    # the filling factor of the cluster
    filling_factor_samples = np.empty((N_flare, radius_bins))
    unfilled_factor_samples = np.empty((N_flare, radius_bins))
    for i in range(N_flare):
        R_jet_index = np.argmin(np.abs(halo_radius.reshape(radius_bins, 1) - R_outbust_vals[i]))
        R_eq_index = np.argmin(np.abs(halo_radius.reshape(radius_bins, 1) - R_eq_vals[i]))
        for j in range(radius_bins):
            if j <= R_eq_index:
                filling_factor_samples[i, j] = (1 - np.cos(thetaJ))
                unfilled_factor_samples[i, j] = np.cos(thetaJ)
            elif R_eq_index < j <= R_jet_index:
                filling_factor_samples[i, j] = 1 - halo_radius[j]/np.sqrt(halo_radius[j]**2 + (np.tan(thetaFRI)*(halo_radius[j] - R_eq_vals[i] + x_eq_vals[i]))**2)
                unfilled_factor_samples[i, j] = halo_radius[j]/np.sqrt(halo_radius[j]**2 + (np.tan(thetaFRI)*(halo_radius[j] - R_eq_vals[i] + x_eq_vals[i]))**2)
            elif j > R_jet_index:
                filling_factor_samples[i, j] = 0
                unfilled_factor_samples[i, j] = 1
    # calculating the effective energy injection and velocity kick profiles, weighted by their volume contribution
    v_kick_vals = np.empty((N_flare, duty_cycle_res, radius_bins))
    Q_eff_vals = np.empty((N_flare, duty_cycle_res, radius_bins))
    for i in range(N_flare):
        for j in range(duty_cycle_res):
            v_kick_vals[i, j] = np.sqrt(np.add(np.multiply(filling_factor_samples[i], duty_cycle[j]*np.square(heating_v_kick_vals[i])),np.multiply(unfilled_factor_samples[i], np.square(cooling_v_kick))))  # m s^-1
            Q_eff_vals[i, j] = np.sqrt(np.add(np.multiply(filling_factor_samples[i], duty_cycle[j]*np.square(heating_rate_vals[i])), np.multiply(unfilled_factor_samples[i], np.square(cooling_rate))))  # W m^-3
    # calculating the NTP fraction
    NTP_fraction_vals = np.empty((N_flare, duty_cycle_res, radius_bins))
    for i in range(N_flare):
        for j in range(duty_cycle_res):
            NTP_fraction_vals[i, j] = np.divide(np.square(v_kick_vals[i, j]), np.add(np.square(v_kick_vals[i, j]), np.square(th_velocity_profile)))  # %

    ## Including the compact source population in the feedback profiles
    # compact fraction
    f_compact = 0.4
    # number of compact sources to consider
    N_compact = int(np.round(f_compact*N_flare/(1 - f_compact), decimals=0))
    # the cooling-only non-thermal pressure
    cooling_only_NTP_fraction = np.divide(np.square(cooling_v_kick), np.add(np.square(cooling_v_kick), np.square(th_velocity_profile)))  # %

    ## Calculating the mean feedback profiles
    # calculating the mean effective gas velocity kicks (for extended + compact sources)
    mean_v_kick_profile = np.empty((duty_cycle_res, radius_bins))
    mean_Q_eff_profile = np.empty((duty_cycle_res, radius_bins))
    mean_NTP_fraction_profile = np.empty((duty_cycle_res, radius_bins))
    for i in range(duty_cycle_res):
        mean_v_kick_profile[i] = np.mean(np.concatenate((v_kick_vals[:, i], np.tile([cooling_v_kick], (N_compact, 1)))), axis=0)  # m s^-1
        mean_Q_eff_profile[i] = np.mean(np.concatenate((Q_eff_vals[:, i], np.tile([cooling_rate], (N_compact, 1)))), axis=0)  # W m^-3
        mean_NTP_fraction_profile[i] = np.mean(np.concatenate((NTP_fraction_vals[:, i], np.tile([cooling_only_NTP_fraction], (N_compact, 1)))), axis=0)  # %

    ## SAVING THE DATA
    # setting the descriptor for output files
    if len(duty_cycle) == 1:
        descriptor = 'DC_' + str(duty_cycle.item())
    elif len(duty_cycle) > 1:
       descriptor = 'DC_' + str(np.min(duty_cycle)) + '_' + str(np.max(duty_cycle))
    # saving output files
    save_nested_arrays("Q_eff_" + descriptor + ".txt", mean_Q_eff_profile)
    save_nested_arrays("v_kick_" + descriptor + ".txt", mean_v_kick_profile)
    save_nested_arrays("NTP_fraction_" + descriptor + ".txt", mean_NTP_fraction_profile)
