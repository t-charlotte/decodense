#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
prop_tools module (helpers shared by the properties and aap_properties modules)
"""

import numpy as np
from pyscf import gto, scf, dft, lib
from pyscf.dft import numint
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.dft import numint as pbc_numint
from typing import Union, Optional, Iterable, Iterator

from .pbctools import get_nuc_pbc
from .tools import dim, contract, logger

# atom-wise (array) or orbital-wise (list of arrays per spin) xc energies
XCPart = Union[np.ndarray, list[np.ndarray]]


def _h_core(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function returns the components of the core hamiltonian
    """
    if not isinstance(mol, pbc_gto.Cell) and mol.has_ecp():
        raise NotImplementedError("Decodense does not support ECP")
    if isinstance(mol, pbc_gto.Cell) and isinstance(
        mf, (pbc_scf.hf.RHF, pbc_scf.uhf.UHF)
    ):
        # kinetic integrals
        kin = mol.pbc_intor("int1e_kin")
        mydf = mf.with_df
        # individual atomic potentials
        sub_nuc = get_nuc_pbc(mol, mydf)
    else:
        # kinetic integrals
        kin = mol.intor_symmetric("int1e_kin")
        # individual atomic potentials
        sub_nuc = _get_nuc(mol)
    # total nuclear potential
    nuc = np.sum(sub_nuc, axis=0)
    return kin, nuc, sub_nuc


def _get_nuc(mol: gto.Mole) -> np.ndarray:
    """
    individual atomic potentials for molecules
    """
    # coordinates and charges of nuclei
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    # individual atomic potentials
    sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
    for k in range(mol.natm):
        with mol.with_rinv_origin(coords[k]):
            sub_nuc[k] = -1.0 * mol.intor("int1e_rinv") * charges[k]
    return sub_nuc


def _xc_ao_deriv(xc_func: str) -> tuple[str, int]:
    """
    this function returns the type of xc functional and the level of ao derivatives
    needed
    """
    xc_type = dft.libxc.xc_type(xc_func)
    if xc_type in ("LDA", "HF"):
        ao_deriv = 0
    elif xc_type in ("GGA", "NLC"):
        ao_deriv = 1
    elif xc_type == "MGGA":
        ao_deriv = 2
    else:
        raise NotImplementedError(f"xc functional type {xc_type} not supported")
    return xc_type, ao_deriv


def _make_rho_interm1(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    this function returns the rho intermediates (c0, c1) needed in _make_rho()
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # compute rho intermediate based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        c0 = contract("ik,kj->ij", ao_value, rdm1)
        c1 = None
    elif xctype in ("GGA", "NLC"):
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = None
    else:  # meta-GGA
        ngrids, nao = ao_value[0].shape
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = np.empty((3, ngrids, nao), dtype=np.float64)
        for i in range(1, 4):
            c1[i - 1] = contract("ik,kj->ij", ao_value[i], rdm1)
    return c0, c1


def _make_rho_interm2(
    c0: np.ndarray, c1: Optional[np.ndarray], ao_value: np.ndarray, xc_type: str
) -> np.ndarray:
    """
    this function returns rho from intermediates (c0, c1)
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # determine dimensions based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        ngrids = ao_value.shape[0]
    else:
        ngrids = ao_value[0].shape[0]
    # compute rho intermediate based on xctype
    if xctype == "LDA" or xctype == "HF":
        rho = contract("pi,pi->p", ao_value, c0)
    elif xctype in ("GGA", "NLC"):
        rho = np.empty((4, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", c0, ao_value[0])
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
    else:  # meta-GGA
        assert c1 is not None
        rho = np.empty((6, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", ao_value[0], c0)
        rho[5] = 0.0
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
            rho[5] += contract("pi,pi->p", c1[i - 1], ao_value[i])
        XX, YY, ZZ = 4, 7, 9
        ao_value_2 = ao_value[XX] + ao_value[YY] + ao_value[ZZ]
        rho[4] = contract("pi,pi->p", c0, ao_value_2)
        rho[4] += rho[5]
        rho[4] *= 2.0
        rho[5] *= 0.5
    return rho


def _make_rho(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    this function returns important dft intermediates, e.g., energy density, grid
    weights, etc.
    """
    # rho corresponding to given 1-RDM
    if rdm1.ndim == 2:
        c0, c1 = _make_rho_interm1(ao_value, rdm1, xc_type)
        rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
    else:
        if np.allclose(rdm1[0], rdm1[1]):
            c0, c1 = _make_rho_interm1(ao_value, rdm1[0] * 2.0, xc_type)
            rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
        else:
            c0_a, c1_a = _make_rho_interm1(ao_value, rdm1[0], xc_type)
            c0_b, c1_b = _make_rho_interm1(ao_value, rdm1[1], xc_type)
            rho = np.stack(
                (
                    _make_rho_interm2(c0_a, c1_a, ao_value, xc_type),
                    _make_rho_interm2(c0_b, c1_b, ao_value, xc_type),
                )
            )
            c0 = c0_a + c0_b
            if c1_a is not None and c1_b is not None:
                c1 = c1_a + c1_b
            else:
                c1 = None
    return c0, c1, rho


def _vk_dft(
    mol: gto.Mole,
    mf: dft.rks.KohnShamDFT,
    xc_func: str,
    rdm1: np.ndarray,
    vk: Optional[np.ndarray],
) -> np.ndarray:
    """
    this function returns the appropriate dft exchange operator
    (vk: unscaled exchange operator if already available, otherwise None)
    """
    # range-separated and exact exchange parameters
    ks_omega, ks_alpha, ks_hyb = mf._numint.rsh_and_hybrid_coeff(xc_func)
    # if hybrid func: compute vk
    if abs(ks_hyb) > 1e-10:
        if vk is None:
            vk = mf.get_k(mol, rdm1)
        # scale amount of exact exchange
        vk *= ks_hyb
    else:
        vk = np.zeros_like(rdm1)  # rdm1 has the same layout as vk
    # range separated coulomb operator
    if abs(ks_omega) > 1e-10:
        vk_lr = mf.get_k(mol, rdm1, omega=ks_omega)
        vk_lr *= ks_alpha - ks_hyb
        vk += vk_lr  # vk_lr has the same (restricted or unrestricted) layout as vk
    return vk


def _ao_val(mol: gto.Mole, grids_coords: np.ndarray, ao_deriv: int) -> np.ndarray:
    """
    this function returns ao function values on the given grid
    """
    if not isinstance(mol, pbc_gto.Cell):
        return numint.eval_ao(mol, grids_coords, deriv=ao_deriv)
    else:
        return pbc_numint.eval_ao(mol, grids_coords, deriv=ao_deriv)


def _orb_atom_occ(
    mo_occ: tuple[np.ndarray, np.ndarray],
    weights: list[np.ndarray],
    spin_mos: tuple[np.ndarray, np.ndarray],
    natm: int,
) -> list[np.ndarray]:
    """
    this function returns the orbital-to-atom occupation factors F[i] - shape
    (nocc_i, natm) - such that rdm1_atom[i,a] = sum_m F[i][m,a] |c_m><c_m|
    """
    F = []
    for i, spin_mo in enumerate(spin_mos):
        if len(spin_mo) == 0:
            F.append(np.zeros((0, natm), dtype=np.float64))
            continue
        w = np.asarray(weights[i], dtype=np.float64)
        F.append(mo_occ[i][spin_mo][:, None] * w / np.sum(w, axis=1, keepdims=True))
    return F


def _xc_blocks(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: dft.rks.KohnShamDFT,
    rdm1: np.ndarray,
    xc_type: str,
    ao_deriv: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    this function yields ao values (without derivatives), weighted xc energy
    densities, w(r) eps_xc(r), and the rho intermediate c0 = ao . D (D summed over
    spins), block-wise on the dft grid (for molecules, ao values are never stored for
    the full grid)
    """
    if xc_type == "HF":
        return
    if mf.grids.coords is None or mf.grids.weights is None:
        logger.warning(
            f'\nCalculating grids for mf.xc="{mf.xc}" with mf.grids.level={mf.grids.level}\n'
            "Should you wish to use other values, please set mf.xc and mf.grids.level in your input.\n"
        )
        mf.grids.build(with_non0tab=True)
    if isinstance(mol, pbc_gto.Cell):
        # pbc block_loop has a different interface -> single block
        blocks: Iterable = [(_ao_val(mol, mf.grids.coords, ao_deriv), mf.grids.weights)]
    else:
        max_memory = max(2000, mf.max_memory - lib.current_memory()[0])
        blocks = (
            (ao, weight)
            for ao, _, weight, _ in mf._numint.block_loop(
                mol, mf.grids, mol.nao_nr(), ao_deriv, max_memory=max_memory
            )
        )
    xc_spin = 0 if np.allclose(rdm1[0], rdm1[1]) else 1
    for ao, weight in blocks:
        c0, c1, rho = _make_rho(ao, rdm1, xc_type)
        # c1 is only needed for rho
        del c1
        eps_xc = dft.libxc.eval_xc(mf.xc, rho, spin=xc_spin, deriv=0)[0]
        yield (ao if ao.ndim == 2 else ao[0]), weight * eps_xc, c0
        # free c0 before the next block is evaluated
        del c0


def _nlc_blocks(
    mol: gto.Mole, mf: dft.rks.KohnShamDFT, rdm1: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    this function returns ao values (without derivatives), weighted nlc (vv10)
    energy densities, w(r) eps_nlc(r), and the rho intermediate c0 = ao . D on the nlc
    grid
    (adapted from: rks.get_veff and numint.nr_nlc_vxc in PySCF)
    """
    _build_grids(mf.nlcgrids)
    nlc_xc = mf.xc if mf._numint.libxc.is_nlc(mf.xc) else mf.nlc
    coords, weights = mf.nlcgrids.coords, mf.nlcgrids.weights
    ao_value_nlc = _ao_val(mol, coords, 1)
    c0_vv10, _, rho_vv10 = _make_rho(ao_value_nlc, np.sum(rdm1, axis=0), "GGA")
    eps_xc_nlc = sum(
        fac * numint._vv10nlc(rho_vv10, coords, rho_vv10, weights, coords, nlc_pars)[0]
        for nlc_pars, fac in mf._numint.nlc_coeff(nlc_xc)
    )
    return [(ao_value_nlc[0], weights * eps_xc_nlc, c0_vv10)]


def _xc_partition(
    blocks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    mol: Union[gto.Mole, pbc_gto.Cell],
    mo_coeff: tuple[np.ndarray, np.ndarray],
    mo_occ: tuple[np.ndarray, np.ndarray],
    weights: Optional[list[np.ndarray]],
    part_method: Optional[str],
    natm: int,
) -> XCPart:
    """
    this function partitions an xc energy, E = sum_r w(r) eps(r) rho(r), given
    block-wise as ao values, weighted energy densities and c0 = ao . D, into
    contributions from individual atoms ("mo"/"ao") or orbitals (otherwise)
    """
    spin_mos = dim(mo_occ)
    if part_method == "ao":
        # ao contributions: e_mu = sum_r w(r) eps(r) chi_mu(r) sum_nu D_munu chi_nu(r)
        # (sum_nu D_munu chi_nu(r) = c0, reused from the evaluation of rho)
        e_ao = np.zeros(mol.nao_nr(), dtype=np.float64)
        for ao0, weps, c0 in blocks:
            e_ao += contract("p,pi,pi->i", weps, ao0, c0)
            # free c0 before the next block is evaluated
            del c0
        ao_atm = np.array([lbl[0] for lbl in mol.ao_labels(fmt=None)])
        return np.bincount(ao_atm, weights=e_ao, minlength=natm)
    # orbital contributions: e_m = sum_r w(r) eps(r) |phi_m(r)|^2
    e_orb = [np.zeros(spin_mo.size, dtype=np.float64) for spin_mo in spin_mos]
    for ao0, weps, c0 in blocks:
        for i, spin_mo in enumerate(spin_mos):
            e_orb[i] += weps @ (ao0 @ mo_coeff[i][:, spin_mo]) ** 2
        # free c0 before the next block is evaluated
        del c0
    if part_method == "mo":
        # atom contributions via orbital-to-atom occupation factors
        assert weights is not None
        F = _orb_atom_occ(mo_occ, weights, spin_mos, natm)
        return sum(e_orb[i] @ F[i] for i in range(2))
    # occupation-weighted orbital contributions (indexed by mo index)
    e_mo = [np.zeros(mo_occ[i].size, dtype=np.float64) for i in range(2)]
    for i, spin_mo in enumerate(spin_mos):
        e_mo[i][spin_mo] = mo_occ[i][spin_mo] * e_orb[i]
    return e_mo


def _xc_energies(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: dft.rks.KohnShamDFT,
    mo_coeff: tuple[np.ndarray, np.ndarray],
    mo_occ: tuple[np.ndarray, np.ndarray],
    rdm1: np.ndarray,
    weights: Optional[list[np.ndarray]],
    part_method: Optional[str],
    natm: int,
) -> tuple[XCPart, Optional[XCPart]]:
    """
    this function returns the "pure" xc and nlc (vv10) energies (None if no nlc) from
    individual atoms ("mo"/"ao") or orbitals (otherwise)
    """
    xc_type, ao_deriv = _xc_ao_deriv(mf.xc)
    e_xc = _xc_partition(
        _xc_blocks(mol, mf, rdm1, xc_type, ao_deriv),
        mol,
        mo_coeff,
        mo_occ,
        weights,
        part_method,
        natm,
    )
    e_xc_nlc = None
    if not isinstance(mol, pbc_gto.Cell) and mf.do_nlc():
        e_xc_nlc = _xc_partition(
            _nlc_blocks(mol, mf, rdm1),
            mol,
            mo_coeff,
            mo_occ,
            weights,
            part_method,
            natm,
        )
    return e_xc, e_xc_nlc
