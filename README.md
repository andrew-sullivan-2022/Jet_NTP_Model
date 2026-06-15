**Function for calculating ensemble-average jet heating:**
-----------------------------------------

**calculate_ensemble_jet_heating**(**log_Qjet_vals**,   **log_age_vals**,   **duty_cycle**,   **redshift**,   **gas_density_profile**,   **temperature_profile**,   **halo_radius**,   **log_dt**=_0.01_)

________________________
__Parameters:__


**_log_Qjet_vals_ : float or array-like**


Logarithmic jet power [log W]


**_log_age_vals_ : float or array-like**


Logarithmic source age age [log yr]


**_duty_cycle_ : float or array-like**


Duty cycle of the AGN [percent]


**_gas_density_profile_ : array-like**

  Gas density [kg/m^3] of the environment, with values corresponding to _halo_radius_


**_temperature_profile_ : array-like**

  Temperature [K] of the environment, with values correspond to _halo_radius_


**_halo_radius_ : array-like**

  Radial component [m] of _gas_density_profile_ and _temperature_profile_

________________________
__Returns:__


Creates array files (.txt) for:

**_Q_eff_ : array-like**

  Effective radially-averaged volumetric power [W/m^3] of the AGN


**_v_kick_ : array-like**

  Velocity kick [m/s] imparted on the gas


**_NTP_fraction_ : array-like**

  Fraction of non-thermal pressure to total pressure [percent] 

______________________________________________________________________________________
