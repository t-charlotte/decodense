#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
schemes module
"""

import numpy as np
from pyscf.pbc import gto as pbc_gto

from .aap_properties import aap_prop_tot, a2ap_redistribute
from .bonds import bond_mbo
from .decomp import CompKeys
from .orbitals import assign_rdm1s
from .properties import prop_tot
from .tools import write_rdm1, logger

from typing import Optional

# minimal verbosity for which the intermediates of the bond-wise schemes are returned
VERBOSE_INTERMEDIATES = 1


def _scheme_atoms(
    mol,
    mf,
    mo_coeff,
    mo_occ,
    rdm1,
    decomp,
):
    """
    This function takes care of atom-wise decompositions:
    1. Compute atomic weights (only if part_method == "mo")
    2. Perform the decomposition
    3. Write the RDM1s if requested
    """

    if decomp.part == "bonds":
        part = "atoms"
        part_method = "mo"
    else:
        part = "atoms"
        part_method = "ao"

    # 1. Compute atomic weights
    if part_method == "mo":
        weights = assign_rdm1s(
            mol,
            mf,
            mo_coeff,
            mo_occ,
            decomp.minao,
            decomp.pop_method,
            decomp.ndo,
            decomp.verbose,
        )
    else:
        weights = None

    # 2. Perform the decomposition
    res = prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        decomp.prop,
        part_method,
        decomp.ndo,
        decomp.gauge_origin,
        weights,
        decomp.atom_ref,
    )

    # 3. Write the RDM1s if requested
    if decomp.write != "":
        write_rdm1(
            mol, part, mo_coeff, mo_occ, decomp.write, decomp.writename, weights
        )

    return res


# end _scheme_atoms


def _scheme_orbitals(mol, mf, mo_coeff, mo_occ, rdm1, decomp):
    """
    This function takes care of orbital-wise decompositions.
    """
    res = prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        decomp.prop,
        None,  # decomp.part_method
        decomp.ndo,
        decomp.gauge_origin,
        weights=None,
    )

    # Write the RDM1s if requested
    if decomp.write != "":
        write_rdm1(
            mol, decomp.part, mo_coeff, mo_occ, decomp.write, decomp.writename, weights
        )

    return res


# end _scheme_orbitals


def _add_intermediates(decomp, atom_res, atom_ref):
    """
    This function stores the intermediates of the bond-wise decomposition
    (atom-wise for a2b, atom-and-atom-pair-wise for aap2b) in decomp.res_inter
    (printed as a separate results object)
    """
    e_tot = atom_res[CompKeys.tot]
    res_inter = { CompKeys.tot : e_tot }

    if atom_ref is not None:
        # isolated-atom energies only exist for atoms: zeros for the atom pairs (aap2b)
        res_inter[CompKeys.atom_ref] = np.concatenate((atom_ref, np.zeros(e_tot.size - atom_ref.size)))

    decomp.res_inter = res_inter


# end _add_intermediates


def _scheme_bonds_a2b(mol, mf, mo_coeff, mo_occ, rdm1, decomp):
    """
    This function takes care of bond-wise decompositions
    using the atoms-to-bonds scheme:
    1. Compute isolated-atom energies
    2. Perform an atom-wise decomposition
    3. Compute bond weights
    4. Perform the bond-wise decomposition
    """

    # unsupported options
    if decomp.prop != "energy":
        raise NotImplementedError("Bond-wise decomposition of dipoles NYI!")
    if decomp.ndo:
        raise NotImplementedError("Bond-wise decomposition for NDOs NYI!")

    # 2. Perform an atom-wise decomposition
    atom_res = _scheme_atoms(  # NOTE: MO for now -> how to implement choosing AO?
        mol, mf, mo_coeff, mo_occ, rdm1, decomp
    )
    logger.warning(
        'The "a2b" bond-wise decomposition scheme uses '
        "Eriksen's MO-based atom-wise decomposition scheme."
    )

    # 3. Compute bond weights
    bond_weights = bond_mbo(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        False, # decomp.ndo
        decomp.verbose,
        decomp.bond_mask,
    )

    # 4. Perform the bond-wise decomposition
    bond_res = a2ap_redistribute(atom_res, bond_weights, decomp.bond_mask, decomp.atom_ref, aap=False)
    if decomp.verbose >= VERBOSE_INTERMEDIATES:
        _add_intermediates(decomp, atom_res, decomp.atom_ref)
    return bond_res


# end _scheme_bonds_a2b


def _scheme_bonds_aap2b(mol, mf, mo_coeff, mo_occ, rdm1, decomp):
    """
    This function takes care of bond-wise decompositions
    using the atoms-and-atom-pairs-to-bonds scheme:
    1. Compute atomic weights
    2. Perform an atom-and-atom-pair-wise decomposition
    3. Compute bond weights
    4. Compute isolated-atom energies
    5. Perform the bond-wise decomposition
    """

    # unsupported options
    if isinstance(mol, pbc_gto.Cell):
        raise NotImplementedError("AAP2B bond-wise decomposition for periodic systems NYI!")
    if decomp.prop != "energy":
        raise NotImplementedError("Bond-wise decomposition of dipoles NYI!")
    if decomp.ndo:
        raise NotImplementedError("Bond-wise decomposition for NDOs NYI!")
    if any(hasattr(mf, attr) for attr in ("mm_mol", "with_solvent", "h1e_mmpol")):
        raise NotImplementedError("AAP2B bond-decomposition for solvation energies NYI!")

    # 1. Compute atomic weights
    weights = assign_rdm1s(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        False,  # ndo
        decomp.verbose,
    )

    # 2. Perform an atom-and-atom-pair-wise decomposition
    aap_res = aap_prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        weights,
    )

    # 3. Compute bond weights
    aap2b_weights = bond_mbo(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        False, # decomp.ndo
        decomp.verbose,
        decomp.bond_mask,
    )

    # 5. Perform the bond-wise decomposition
    bond_res = a2ap_redistribute(aap_res, aap2b_weights, decomp.bond_mask, decomp.atom_ref, aap=True)
    if decomp.verbose >= VERBOSE_INTERMEDIATES:
        _add_intermediates(decomp, aap_res, decomp.atom_ref)
    return bond_res


# end _scheme_bonds_aap2b

SCHEMES = {
    ("atoms", "mo"): _scheme_atoms,
    ("atoms", "ao"): _scheme_atoms,
    ("orbitals", None): _scheme_orbitals,
    ("bonds", "a2b"): _scheme_bonds_a2b,
    ("bonds", "aap2b"): _scheme_bonds_aap2b,
}
