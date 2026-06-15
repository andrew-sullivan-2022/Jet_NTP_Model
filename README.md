Function for calculating ensemble-average jet heating:

calculate_AGN_heating(log_Qjet_vals, log_active_age_vals, duty_cycle, gas_density_profile, temperature_profile, halo_radius)

Parameters:

log_Qjet_vals : float or array-like

Logarithmic jet power [log W]

log_active_age_vals : float or array-like

Logarithmic active age [log yr]

duty_cycle : float or array-like

Duty cycle of the AGN [percent]

gas_density_profile : array-like

Gas density [kg/m^3] of the environment, with values corresponding to halo_radius

temperature_profile : array-like

Temperature [K] of the environment, with values correspond to halo_radius

halo_radius : array-like

Radial component [m] of gas_density_profile and temperature_profile

Returns:

Creates array files (.txt) for:

Q_eff : array-like

Effective radially-averaged volumetric power [W/m^3] of the AGN

v_kick : array-like

Velocity kick [m/s] imparted on the gas

NTP_fraction : array-like

Fraction of non-thermal pressure to total pressure [percent]
