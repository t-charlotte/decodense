#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
schemes module
"""

import numpy as np
from pyscf.pbc import gto as pbc_gto

from .aap_properties import aap_prop_tot, a2ap_redistribute
from .atoms import atom_ref_energies
from .bonds import bond_mbo
from .decomp import CompKeys
from .orbitals import assign_rdm1s
from .properties import prop_tot
from .tools import write_rdm1, logger

# minimal verbosity for which the intermediates of the bond-wise schemes are returned
VERBOSE_INTERMEDIATES = 1


def _scheme_atoms_mo(mol, mf, mo_coeff, mo_occ, rdm1, decomp):
    """
    This function takes care of MO-based atom-wise decompositions:
    1. Compute atomic weights
    2. Perform the decomposition
    3. Writes the RDM1s if requested
    """
    # 1. Compute atomic weights
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
        "mo", # decomp.part_method
        decomp.ndo,
        decomp.gauge_origin,
        weights,
    )
    # 3. Writes the RDM1s if requested
    if decomp.write != "":
        write_rdm1(
            mol, decomp.part, mo_coeff, mo_occ, decomp.write, decomp.writename, weights
        )

    return res


# end _scheme_atoms


def _scheme_atoms_ao_orbitals(mol, mf, mo_coeff, mo_occ, rdm1, decomp):
    """
    This function takes care of AO-based atom-wise decompositions
    and orbital-wise decompositions.
    The difference between these two is indicated by decomp.part_method.
    """
    return prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        decomp.prop,
        decomp.part_method,
        decomp.ndo,
        decomp.gauge_origin,
        weights=None,
    )


# end _scheme_orbitals


def _atom_ref(mol, mf):
    """
    This function returns the gas-phase isolated-atom energies
    """
    if any(hasattr(mf, attr) for attr in ("with_solvent", "mm_mol", "h1e_mmpol")):
        logger.warning(
            "Warning: the molecular energy includes solvation, but the isolated-atom "
            f'energies are calculated in the gas phase ("{CompKeys.tot_rel}").'
        )
    return atom_ref_energies(mol, mf)


# end _atom_ref


def _add_intermediates(decomp, atom_res, atom_ref):
    """
    This function stores the intermediates of the bond-wise decomposition
    (atom-wise for a2b, atom-and-atom-pair-wise for aap2b) in decomp.res_inter
    (printed as a separate results object)
    """
    e_tot = atom_res[CompKeys.tot]
    # isolated-atom energies only exist for atoms: zeros for the atom pairs (aap2b)
    atom_ref = np.concatenate((atom_ref, np.zeros(e_tot.size - atom_ref.size)))
    decomp.res_inter = {
        CompKeys.tot: e_tot,
        CompKeys.atom_ref: atom_ref,
    }


# end _add_intermediates

def _scheme_bonds_a2b(mol, mf, mo_coeff, mo_occ, rdm1, decomp):
    """
    This function takes care of bond-wise decompositions
    using the atoms-to-bonds scheme:
    1. Perform an atom-wise decomposition
    2. Compute bond weights
    3. Compute isolated-atom energies
    4. Perform the bond-wise decomposition
    """

    # unsupported options
    if decomp.prop != "energy":
        raise NotImplementedError("Bond-wise decomposition of dipoles NYI!")

    # 1. Perform an atom-wise decomposition
    atom_res = _scheme_atoms_mo(  # NOTE: MO for now -> how to implement choosing AO?
        mol, mf, mo_coeff, mo_occ, rdm1, decomp
    )
    logger.warning(
        'The "a2b" bond-wise decomposition scheme uses '
        'Eriksen\'s MO-based atom-wise decomposition scheme.'
    )

    # 2. Compute bond weights
    bond_weights, is_bond = bond_mbo(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        decomp.ndo,
        decomp.verbose,
        decomp.bond_crit,
        decomp.mbo_thresh,
        decomp.smiles,
        decomp.lewis_image,
        decomp.trust_atom_order
    )

    # 3. Compute isolated-atom energies
    atom_ref = _atom_ref(mol, mf)

    # 4. Perform the bond-wise decomposition
    bond_res = a2ap_redistribute(
        atom_res, bond_weights, is_bond, atom_ref, aap=False
    )
    if decomp.verbose >= VERBOSE_INTERMEDIATES:
        _add_intermediates(decomp, atom_res, atom_ref)
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
        raise NotImplementedError("AAP decomposition for periodic systems NYI!")
    if decomp.prop != "energy":
        raise NotImplementedError("AAP decomposition of dipoles NYI!")
    if decomp.ndo:
        raise NotImplementedError("AAP decomposition for NDOs NYI!")
    if any(hasattr(mf, attr) for attr in ("mm_mol", "with_solvent", "h1e_mmpol")):
        raise NotImplementedError("AAP decomposition for solvation energies NYI!")

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
    aap2b_weights, is_bond = bond_mbo(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        decomp.ndo,
        decomp.verbose,
        decomp.bond_crit,
        decomp.mbo_thresh,
        decomp.smiles,
        decomp.lewis_image,
        decomp.trust_atom_order
    )

    # 4. Compute isolated-atom energies
    atom_ref = _atom_ref(mol, mf)

    # 5. Perform the bond-wise decomposition
    bond_res = a2ap_redistribute(
        aap_res, aap2b_weights, is_bond, atom_ref, aap=True
    )
    if decomp.verbose >= VERBOSE_INTERMEDIATES:
        _add_intermediates(decomp, aap_res, atom_ref)
    return bond_res


# end _scheme_bonds_aap2b

SCHEMES = {
    ("atoms", "mo"): _scheme_atoms_mo,
    ("atoms", "ao"): _scheme_atoms_ao_orbitals,
    ("orbitals", None): _scheme_atoms_ao_orbitals,
    ("bonds", "a2b"): _scheme_bonds_a2b,
    ("bonds", "aap2b"): _scheme_bonds_aap2b,
}
