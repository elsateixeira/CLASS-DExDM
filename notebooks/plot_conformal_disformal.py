#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from classy import Class


def run_class(params):
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo


def cleanup(cosmo):
    try:
        cosmo.struct_cleanup()
    finally:
        cosmo.empty()


common = {
    'omega_b': 2.255065e-02,
    'h': 0.67810,
    'A_s': 2.123257e-09,
    'n_s': 0.9665,
    'tau_reio': 0.0543,
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_h/Mpc': 10.0,
    'z_pk': '0,1,3',
    'k_output_values': '0.05',
}

lcdm_params = dict(common)
lcdm_params.update({'omega_cdm': 1.193524e-01})

conformal_params = dict(common)
conformal_params.update({
    'omega_cdm': 1e-10,
    'omega_idm': 1.193524e-01,
    'Omega_scf': -1,
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'scf_potential': 'exp',
    'scf_coupling_type': 'conformal',
    'scf_V0': 1.0,
    'scf_lambda': 0.1,
    'scf_beta': -0.05,
    'scf_C0': 1.0,
    'attractor_ic_scf': 'no',
    'scf_phi_ini': 1.0,
    'scf_phi_prime_ini': -1e-8,
    'scf_shooting_target': 'scf_V0',
})

disformal_params = dict(common)
disformal_params.update({
    'omega_cdm': 1e-10,
    'omega_idm': 1.193524e-01,
    'Omega_scf': -1,
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'scf_potential': 'exp',
    'scf_coupling_type': 'disformal',
    'scf_V0': 1.0,
    'scf_lambda': 1.0,
    'scf_alpha': 0.0,
    'scf_D0': 2.0,
    'attractor_ic_scf': 'no',
    'scf_phi_ini': 1.0,
    'scf_phi_prime_ini': 0.0,
    'scf_shooting_target': 'scf_V0',
})

out_dir = os.path.join('notebooks', 'figures')
os.makedirs(out_dir, exist_ok=True)

lcdm = conf = disf = None
try:
    lcdm = run_class(lcdm_params)
    conf = run_class(conformal_params)
    disf = run_class(disformal_params)

    bg_lcdm = lcdm.get_background()
    bg_conf = conf.get_background()
    bg_disf = disf.get_background()

    z = bg_lcdm['z']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(1 + z, bg_lcdm['H [1/Mpc]'], lw=2, label='LCDM')
    ax.loglog(1 + z, bg_conf['H [1/Mpc]'], lw=2, ls='--', label='Conformal')
    ax.loglog(1 + z, bg_disf['H [1/Mpc]'], lw=2, ls=':', label='Disformal')
    ax.set_xlabel(r'$1+z$')
    ax.set_ylabel(r'$H\,[1/\mathrm{Mpc}]$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'coupling_Hz_conformal_vs_disformal.png'), dpi=180)
    plt.close(fig)

    cls_lcdm = lcdm.lensed_cl(2500)
    cls_conf = conf.lensed_cl(2500)
    cls_disf = disf.lensed_cl(2500)

    ell = cls_lcdm['ell'][2:]
    pref = ell * (ell + 1) / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('log')
    ax.plot(ell, cls_lcdm['tt'][2:] * pref, lw=2, label='LCDM')
    ax.plot(ell, cls_conf['tt'][2:] * pref, lw=2, ls='--', label='Conformal')
    ax.plot(ell, cls_disf['tt'][2:] * pref, lw=2, ls=':', label='Disformal')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'coupling_CMB_TT_conformal_vs_disformal.png'), dpi=180)
    plt.close(fig)

    k = np.logspace(-4, np.log10(common['P_k_max_h/Mpc'] * 0.95), 220)
    h = common['h']
    pk_lcdm = np.array([lcdm.pk(ki * h, 0.0) * h**3 for ki in k])
    pk_conf = np.array([conf.pk(ki * h, 0.0) * h**3 for ki in k])
    pk_disf = np.array([disf.pk(ki * h, 0.0) * h**3 for ki in k])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k, pk_lcdm, lw=2, label='LCDM')
    ax.loglog(k, pk_conf, lw=2, ls='--', label='Conformal')
    ax.loglog(k, pk_disf, lw=2, ls=':', label='Disformal')
    ax.set_xlabel(r'$k\,[h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$P(k)\,[\mathrm{(Mpc}/h)^3]$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'coupling_Pk_conformal_vs_disformal.png'), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(k, pk_conf / pk_lcdm, lw=2, ls='--', label='Conformal/LCDM')
    ax.semilogx(k, pk_disf / pk_lcdm, lw=2, ls=':', label='Disformal/LCDM')
    ax.axhline(1.0, color='k', lw=1)
    ax.set_xlabel(r'$k\,[h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$P(k)/P_{\Lambda\mathrm{CDM}}$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'coupling_Pk_ratio_conformal_vs_disformal.png'), dpi=180)
    plt.close(fig)

    print('Saved plots in', out_dir)

finally:
    for obj in (disf, conf, lcdm):
        if obj is not None:
            cleanup(obj)
