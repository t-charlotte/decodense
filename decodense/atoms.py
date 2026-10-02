#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
atoms module
"""

import numpy as np
from pyscf import gto, scf, dft
from pyscf.data import elements
from typing import Union, Any

from .tools import logger

# max. number of stability-analysis cycles in the isolated-atom calculations
MAX_STAB_CYCLES = 5

# min. energy lowering (Eh) for following an instability further;
# smaller changes are numerical noise (e.g. rotations within a degenerate open shell)
STAB_E_TOL = 1.0e-5

# calculation settings copied from the molecular calculation (if present)
SCF_ATTRS = ("conv_tol", "conv_tol_grad", "max_cycle", "max_memory")
DFT_ATTRS = ("xc", "nlc", "small_rho_cutoff")
GRID_ATTRS = (
    "level",
    "atom_grid",
    "prune",
    "radi_method",
    "becke_scheme",
    "radii_adjust",
    "atomic_radii",
)


def atom_ref_energies(
    mol: gto.Mole,
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT],
) -> np.ndarray:
    """
    this function returns the total energies of the isolated, neutral atoms (natm,)
    in the gas phase and in their Hund's-rule ground state, computed with UHF/UKS
    using the same basis set and calculation settings as mf
    """
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)
    e_ref = np.zeros(mol.natm, dtype=np.float64)
    # one calculation per atom label (labels may carry their own basis, e.g. "C1")
    cache: dict[str, float] = {}
    for i in range(mol.natm):
        label = mol.atom_symbol(i)
        if label not in cache:
            cache[label] = _e_atom(mol, mf, i, dft_calc)
        e_ref[i] = cache[label]
    return e_ref


# end def atom_ref_energies()


def _hund_spin(z: int) -> int:
    """
    this function returns 2S (the number of unpaired electrons) of the neutral atom
    with nuclear charge z, according to its ground-state configuration and Hund's rule
    """
    n_unpaired = 0
    for l, n_el in enumerate(elements.CONFIGURATION[z]):  # [s, p, d, f] electron counts
        cap = 2 * (2 * l + 1)
        n_open = n_el % cap
        n_unpaired += min(n_open, cap - n_open)
    return n_unpaired


# end def _hund_spin()


def _copy_attrs(dst: Any, src: Any, attrs: tuple[str, ...]) -> None:
    """
    this function copies the given attributes from src to dst (if present in src)
    """
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


# end def _copy_attrs()


def _e_atom(
    mol: gto.Mole,
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT],
    atom_idx: int,
    dft_calc: bool,
) -> float:
    """
    this function returns the UHF/UKS total energy of an isolated, neutral atom
    """
    # ghost atoms have no nucleus and no electrons
    if mol.atom_charge(atom_idx) == 0:
        return 0.0

    label = mol.atom_symbol(atom_idx)
    z = elements.charge(mol.atom_pure_symbol(atom_idx))

    # isolated atom with the same basis set as in the molecule
    atm = gto.M(
        atom=[[label, (0.0, 0.0, 0.0)]],
        basis={label: mol._basis[label]},
        charge=0,
        spin=_hund_spin(z),
        cart=mol.cart,
        symmetry=False,
        max_memory=mol.max_memory,
        verbose=0,
    )

    # same method and calculation settings as for the molecule
    mf_atm = dft.UKS(atm) if dft_calc else scf.UHF(atm)
    _copy_attrs(mf_atm, mf, SCF_ATTRS)
    if dft_calc:
        _copy_attrs(mf_atm, mf, DFT_ATTRS)
        # range-separation parameter (None: taken from the functional)
        if getattr(mf, "omega", None) is not None:
            mf_atm.omega = mf.omega
        _copy_attrs(mf_atm.grids, mf.grids, GRID_ATTRS)
        _copy_attrs(mf_atm.nlcgrids, mf.nlcgrids, GRID_ATTRS)
    # density fitting
    if getattr(mf, "with_df", None) is not None:
        mf_atm = mf_atm.density_fit(auxbasis=mf.with_df.auxbasis)

    mf_atm.kernel()
    # lowest energy of a converged solution
    e_best = mf_atm.e_tot if mf_atm.converged else np.inf

    # follow internal instabilities to obtain the lowest-energy UHF/UKS solution
    for _ in range(MAX_STAB_CYCLES):
        mo_new, _, stable, _ = mf_atm.stability(return_status=True)
        if stable:
            break
        mf_atm.kernel(mf_atm.make_rdm1(mo_new, mf_atm.mo_occ))
        # unconverged solutions are not accepted, but are followed further
        if mf_atm.converged:
            if mf_atm.e_tot > e_best - STAB_E_TOL:
                # no significant lowering: the remaining instability is numerical noise
                e_best = min(e_best, mf_atm.e_tot)
                break
            e_best = mf_atm.e_tot
    else:
        logger.warning(
            f"Warning: stability analysis for isolated atom {label} did not finish "
            f"within {MAX_STAB_CYCLES} cycles; its energy may not be the lowest one"
        )

    if not np.isfinite(e_best):
        logger.warning(
            f"Warning: isolated-atom calculation for {label} did not converge"
        )
        return mf_atm.e_tot

    return e_best


# end def _e_atom()
