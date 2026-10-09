#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
decomp module
"""

import numpy as np
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.lib.kpts_helper import gamma_point
from typing import Union, Optional
from .tools import logger


# property components (additive per atom / atom pair / bond); redistributed in bond-wise schemes
class PropKeys:
    coul = "Coul."
    exch = "Exch."
    exch_DFT = "Exch. (DFT)"
    kin = "Kin."
    solvent = "Solv."
    solvent_vdw = "Solv. (vdW)"
    nuc_att_glob = "E_ne (1)"
    nuc_att_loc = "E_ne (2)"
    nuc_att = "E_ne"
    xc = "XC"
    xc_nlc = "XC (nlc)"
    struct = "Struct."
    el = "Elect."
    tot = "Total"
    tot_rel = "Total (rel.)"


# labels, occupations, symmetries and intermediates; never redistributed
class InfoKeys:
    atoms = "Atom"
    atom_pairs = "Atom (pair)"
    orbitals = "Orbital"
    bonds = "Bond"
    mo_occ = "Occup."
    orbsym = "Symm."
    atom_ref = "E_atom_0"
    sum_row = "Sum"


# all component keys; add new keys to PropKeys or InfoKeys, not here
class CompKeys(PropKeys, InfoKeys):
    pass


def _key_values(cls) -> tuple[str, ...]:
    """
    this function returns the values of the keys defined in cls
    """
    return tuple(v for k, v in vars(cls).items() if not k.startswith("__"))


PROP_KEYS = frozenset(_key_values(PropKeys))

# maps every key value to the attribute name used in ResultsCls
comp_key_dict = {
    v: k.lower()
    for cls in (PropKeys, InfoKeys)
    for k, v in vars(cls).items()
    if not k.startswith("__")
}


class DecompCls:
    """
    this class contains all decomp attributes
    """

    __slots__ = (
        "minao",
        "mo_basis",
        "pop_method",
        "mo_init",
        "loc_exp",
        "part",
        "part_method",
        "ndo",
        "gauge_origin",
        "prop",
        "write",
        "writename",
        "verbose",
        "unit",
        "res",
        "res_inter",
        # below: exclusively for bond-wise decomposition schemes
        "bond_crit",
        "mbo_thresh",
        "smiles",
        "lewis_image",
        "trust_atom_order",
    )

    def __init__(
        self,
        minao: str = "MINAO",
        mo_basis: str = "can",
        pop_method: str = "mulliken",
        mo_init: str = "can",
        loc_exp: int = 2,
        part: str = "atoms",
        part_method: Optional[str] = None,
        ndo: bool = False,
        gauge_origin: Optional[np.ndarray] = None,
        prop: str = "energy",
        write: str = "",
        writename: str = "",
        verbose: int = 0,
        unit: str = "au",
        bond_crit: str = "mbo",  # "mbo", "lewis" or "none"
        mbo_thresh: float = 0.8,
        smiles: Optional[str] = None,
        lewis_image: str = "lewis_structure.png",
        trust_atom_order: bool = False,
    ) -> None:
        """
        init molecule attributes
        """
        # set system defaults
        self.minao = minao
        self.mo_basis = mo_basis
        self.pop_method = pop_method
        self.mo_init = mo_init
        self.loc_exp = loc_exp

        if part == "eda":
            logger.warning(
                'Warning: part="eda" is deprecated; use part="atoms", part_method="ao" instead'
            )
            part, part_method = "atoms", "ao"
        elif part == "bonds":
            self.bond_crit = bond_crit
            self.mbo_thresh = mbo_thresh
            self.smiles = smiles
            self.lewis_image = lewis_image
            self.trust_atom_order = trust_atom_order
        # end if
        if part_method is None:
            part_method = {"atoms": "mo"}.get(part)
            # NOTE: sanity_check will raise an error if part == "bonds" and part_method is None
        # end if

        self.part = part
        self.part_method = part_method
        self.ndo = ndo
        self.gauge_origin = (
            np.zeros(3, dtype=np.float64) if gauge_origin is None else gauge_origin
        )
        self.prop = prop
        self.write = write
        self.writename = writename
        self.verbose = verbose
        self.unit = unit
        # set internal defaults
        self.res: dict[str, Union[np.ndarray, list[np.ndarray]]] = {}
        self.res_inter: dict[str, np.ndarray] = {}


def sanity_check(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
    decomp: DecompCls,
    mo_coeff: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    mo_occ: Optional[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]],
):
    """
    this function performs sanity checks of decomp attributes
    """
    # Reference basis for IAOs
    if decomp.minao not in ("MINAO", "ANO"):
        raise ValueError(
            'invalid minao basis. valid choices: "MINAO" (default) or "ANO"'
        )
    # MO basis
    if decomp.mo_basis not in ("can", "fb", "pm"):
        raise ValueError(
            'invalid MO basis. valid choices: "can" (default), "fb", or "pm"'
        )
    # population scheme
    if decomp.pop_method not in ("mulliken", "lowdin", "meta_lowdin", "becke", "iao"):
        raise ValueError(
            'invalid population scheme. valid choices: "mulliken" (default), "lowdin", '
            '"meta_lowdin", "becke", or "iao"'
        )
    # MO start guess (for localization)
    if decomp.mo_init not in ("can", "cholesky", "ibo"):
        raise ValueError(
            'invalid MO start guess. valid choices: "can" (default), "cholesky", or '
            '"ibo"'
        )
    # localization exponent
    if decomp.loc_exp not in (2, 4):
        raise ValueError(
            "invalid localization exponent. valid choices: 2 (default) or 4"
        )
    if decomp.part == "orbitals":
        logger.warning(
            "Warning: This partitioning only computes electronic energy and does not "
            "include solvent van der Waals contributions."
        )
        if decomp.part_method is not None:
            logger.warning(
                "Warning: This partitioning does not require a value for part_method. "
                "The requested partitioning method will be ignored."
            )
            decomp.part_method = None
    elif decomp.part == "atoms":
        if decomp.part_method not in ("ao", "mo"):
            raise ValueError(
                'invalid partitioning method. valid choices for part="atoms": '
                '"mo" (Eriksen\'s MO-based scheme - default) or "ao" (Nakai\'s AO-based energy density analysis scheme)'
            )
    elif decomp.part == "bonds":
        if decomp.part_method not in ("a2b", "aap2b"):
            raise ValueError(
                'invalid partitioning method. valid choices for part="bonds": '
                '"a2b" (atoms-to-bonds) or "aap2b" (atoms-and-atom-pairs-to-bonds)'
            )
        if decomp.pop_method not in ("mulliken", "iao"):
            raise ValueError(
                'invalid population scheme for part="bonds". valid choices: "mulliken" or "iao"'
            )
        if decomp.bond_crit not in ("mbo", "lewis", "none"):
            raise ValueError(
                'invalid bond criterion. valid choices: "mbo" (default), "lewis" or "none"'
            )
        if decomp.bond_crit == "lewis":
            try:
                import rdkit  # noqa: F401
            except ImportError as err:
                raise ImportError('bond criterion "lewis" requires RDKit') from err
        if decomp.bond_crit == "mbo" and (
            decomp.mbo_thresh is None or decomp.mbo_thresh <= 0
        ):
            raise ValueError(
                "Mayer bond order based bond criterion requires a bond order threshold > 0. default value: 0.8"
            )
        if (
            decomp.bond_crit == "lewis"
            and decomp.trust_atom_order
            and decomp.smiles is None
        ):
            raise ValueError("trust_atom_order requires a SMILES string")
    else:
        raise ValueError(
            'invalid partitioning. valid choices: "atoms" (default), "orbitals" or "bonds"'
        )
    # NDO decomposition
    if not isinstance(decomp.ndo, bool):
        raise TypeError("invalid NDO argument. must be a bool")
    # gauge origin
    if not isinstance(decomp.gauge_origin, (list, np.ndarray)):
        raise TypeError(
            "invalid gauge origin. must be a list or numpy array of 3 ints/floats"
        )
    if len(decomp.gauge_origin) != 3 or not all(
        isinstance(coord, (int, float, np.integer, np.floating))
        for coord in decomp.gauge_origin
    ):
        raise ValueError(
            "invalid gauge origin. must be a list or numpy array of 3 ints/floats"
        )
    # property
    if decomp.prop not in ("energy", "dipole"):
        raise ValueError(
            'invalid property. valid choices: "energy" (default) and "dipole"'
        )
    # write
    if not isinstance(decomp.write, str):
        raise TypeError("invalid write format argument. must be a str")
    if not isinstance(decomp.writename, str):
        raise TypeError("invalid write name argument. must be a str")
    if decomp.write not in ("", "cube", "numpy"):
        raise ValueError('invalid write format. valid choices: "cube" and "numpy"')
    if decomp.write != "" and (decomp.part, decomp.part_method) != ("atoms", "mo"):
        raise ValueError('write is only implemented for part="atoms", part_method="mo"')
    # verbosity
    if not isinstance(decomp.verbose, int):
        raise TypeError(
            'invalid verbosity. valid choices: 0 <= "verbose" <= 5 (default: 0)'
        )
    if decomp.verbose < 0 or decomp.verbose > 5:
        raise ValueError(
            'invalid verbosity. valid choices: 0 <= "verbose" <= 5 (default: 0)'
        )
    # cell object
    if isinstance(mol, pbc_gto.Cell):
        if np.shape(mf.kpt) != (3,):
            raise ValueError(
                "PBC module is in development, only gamma-point methods implemented."
            )
        if not gamma_point(mf.kpt):
            raise ValueError(
                "PBC module is in development, only gamma-point methods implemented."
            )
        if mol.dimension != 3 and mol.dimension != 1:
            raise ValueError(
                "PBC module is in development, current implementation treats 1D- and "
                "3D-cells only."
            )
        if decomp.prop != "energy" or decomp.part not in ("atoms", "eda"):
            raise ValueError(
                "PBC module is in development. Only gamma-point calculation of "
                "energy for 1D- and 3D-periodic systems can be decomposed into "
                "atomwise contributions."
            )
    # unit
    if not isinstance(decomp.unit, str):
        raise TypeError(
            'invalid unit. valid choices: "au" (default), "kcal_mol", "ev", '
            '"kj_mol", or "debye"'
        )
    if decomp.unit.lower() not in ("au", "kcal_mol", "ev", "kj_mol", "debye"):
        raise ValueError(
            'invalid unit. valid choices: "au" (default), "kcal_mol", "ev", '
            '"kj_mol", or "debye"'
        )
    # mo coefficients
    if not isinstance(mo_coeff, np.ndarray) and not isinstance(mo_coeff, tuple):
        raise TypeError(
            "invalid mo coefficients. must be a numpy array or tuple of numpy arrays"
        )
    if isinstance(mo_coeff, np.ndarray):
        if mo_coeff.ndim != 2 and mo_coeff.ndim != 3:
            raise ValueError(
                "invalid mo coefficients. must be a numpy array of dimension 2 or 3"
            )
    elif isinstance(mo_coeff, tuple):
        if (
            len(mo_coeff) != 2
            or not isinstance(mo_coeff[0], np.ndarray)
            or not isinstance(mo_coeff[1], np.ndarray)
        ):
            raise TypeError(
                "invalid mo coefficients. must be a tuple of two numpy arrays"
            )
    # mo occupation
    if (
        mo_occ is not None
        and not isinstance(mo_occ, np.ndarray)
        and not isinstance(mo_occ, tuple)
    ):
        raise TypeError(
            "invalid mo occupation. must be a numpy array or tuple of numpy arrays"
        )
    if isinstance(mo_occ, np.ndarray):
        if mo_occ.ndim != 1 and mo_occ.ndim != 2:
            raise ValueError(
                "invalid mo occupation. must be a numpy array of dimension 1 or 2"
            )
    elif isinstance(mo_occ, tuple):
        if (
            len(mo_occ) != 2
            or not isinstance(mo_occ[0], np.ndarray)
            or not isinstance(mo_occ[1], np.ndarray)
        ):
            raise TypeError(
                "invalid mo occupation. must be a tuple of two numpy arrays"
            )
