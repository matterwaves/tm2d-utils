import numpy as np

alpha = 0.0072973525693 # fine structure constant
c = 299792458 # speed of light [m/s]
hbar = 6.62607004e-34/(2*np.pi) # reduced planck's constant [kg*m^2/s]
e_mass = 0.511e6 # electron mass [eV/c^2]
e_compton = 2.42631023867e-2 # compton wavelength [A]
n_avogadro = 6.02214076e23 # avogadro's number [mol^-1]
a0 = 0.529 # bohr radius [A]
e_charge_VA = 14.4 # electron charge [V*A]
e_charge_SI = 1.602176634e-19 # elementary charge [C]
J_per_eV = 1.602176634e-19 # joules per electron volt (unit conversion)
rho_bulk_ice = 0.0314315 # amorphous ice number density [molecules/A^3]

def get_gammaLorentz(HT):
    """
    Calculates Lorentz factor [dimensionless] from HT [V].
    Units: [dimensionless]
    """
    # note electron mass is in [V]
    return 1 + HT / e_mass # [dimensionless]

def get_beta(HT):
    """
    Calculates electron speed [units of speed of light] from HT [V].
    """
    gamma_lorentz = get_gammaLorentz(HT) # Lorentz factor [dimensionless]
    return np.sqrt(1 - 1 / gamma_lorentz**2) # [units of c]

def get_sigmaE(HT):
    """
    Calculates interaction parameter for scaling between projected potential and phase.
    Units: [rad/(V*A)]
    """
    gamma_lorentz = get_gammaLorentz(HT) # Lorentz factor
    beta = get_beta(HT) # electron speed relative to c [dimensionless]
    wlen = e_compton / (gamma_lorentz * beta) # wavelength [A]
    return 2*np.pi / (wlen * HT) * ((e_mass * J_per_eV) + e_charge_SI * HT) /\
        (2 * e_mass * J_per_eV + e_charge_SI*HT) # [rad/(V*A)]

def get_eWlenFromHT(HT):
    """
    Calculates electron wavelength.
    Units: [A]
    """
    return e_compton / (get_gammaLorentz(HT) * get_beta(HT)) # [A]

def get_ghost_spacing(f_OL, HT, wlen_L=1.064e4):
    """
    Calculates LPP ghost spacing from objective focal length [A], HT [V], and laser wavelength [A].
    Units: [A]
    """
    wlen_e = get_eWlenFromHT(HT) # [A] electron wavelength
    return 2 * wlen_e * f_OL / wlen_L # [A]

def dose_A2ToPix(dose_A2, pix_size):
    return dose_A2 * pix_size ** 2 # convert [e/A^2] to [e/pix]

def get_protein_radius(coords):
    coords -= np.mean(coords, axis=0) # subtract center of mass
    return np.sqrt((coords ** 2).sum(axis=1)).max()

def get_random_pose(poses_list):
    return poses_list[np.random.randint(0, poses_list.shape[0]), :] # [deg]