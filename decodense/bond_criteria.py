#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
bond criteria module
"""

import numpy as np

from pyscf import gto
from pyscf.pbc import gto as pbc_gto
from pyscf.data import radii

from .tools import logger

from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING: # only for type annotations, RDKit remains optional
    from rdkit import Chem

def get_bond_mask(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mbo: Optional[np.ndarray],
    bond_crit: str = "none",           # "none", "mbo" or "lewis"
    mbo_thresh: float = 0.1,
    smiles: Optional[str] = None,
    lewis_image: str = "lewis_structure.png",
    trust_atom_order: bool = False,
) -> np.ndarray:
    """
    this function returns a boolean array (ordered according to ap_label) that is True if the
    atom pair is a bond according to bond_crit.
    it requires an array of MBOs (atom_label_0, atom_label_1) as input.

    options:
    - bond_crit = "none": assume all atom pairs are bonds
    - bond_crit = "mbo": an atom pair is a bond if the bond order is above a certain threshold
    - bond_crit = "lewis": an atom pair is a bond if the Lewis structure shows a bond (see lewis_bond_mask)
    
    """
    if bond_crit == "lewis" and isinstance(mol, pbc_gto.Cell):
        raise NotImplementedError('Lewis bond criterion is not implemented for PBC')

    a_idx, b_idx = np.triu_indices(mol.natm, k=1)

    if bond_crit == "mbo":
        is_bond = mbo[a_idx, b_idx] > mbo_thresh
    elif bond_crit == "lewis":
        is_bond = lewis_bond_mask(mol, smiles, lewis_image, mbo, trust_atom_order)
    else:
        is_bond = np.ones(a_idx.size, dtype=bool)

    return is_bond
# end def get_bond_mask()

def _pyscf_to_rdkit(mol: gto.Mole) -> "Chem.Mol":
    """
    this function returns an RDKit molecule (without bond objects)
    with the same atom numbering as the pyscf mol
    """
    from rdkit import Chem

    coords = mol.atom_coords(unit="Angstrom")
    xyz = f"{mol.natm}\n\n" + "\n".join(
        f"{mol.atom_pure_symbol(i)} {x:.10f} {y:.10f} {z:.10f}"
        for i, (x, y, z) in enumerate(coords)
    )
    return Chem.MolFromXYZBlock(xyz)
# end def _pyscf_to_rdkit()

def _save_lewis_image(rdkit_mol: "Chem.Mol", fname: str) -> None:
    """
    this function saves an image of the Lewis structure, with atoms labelled by their pyscf index
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import MolToImage

    rd_draw = Chem.Mol(rdkit_mol) # copy, do not overwrite 3D coordinates
    AllChem.Compute2DCoords(rd_draw) # compute 2D coordinates for making a drawing
    for atom in rd_draw.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    MolToImage(rd_draw, size=(600, 600)).save(fname)
    logger.info(f"Saved image of Lewis structure to \"{fname}\"")
# end def _save_lewis_image()

def _connectivity(mol_smiles: "Chem.Mol") -> "Chem.RWMol":
    """
    this function returns a copy of the RDKit molecule
    that was generated from the SMILES string
    with only its connectivity
    i.e. removing all charges, radicals, isotopes and aromaticity
         and setting every bond to single
    """
    from rdkit import Chem

    connectivity = Chem.RWMol(mol_smiles)
    for atom in connectivity.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsotope(0)
        atom.SetIsAromatic(False)
    for bond in connectivity.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return connectivity
# end def _connectivity()

def _best_match(
    connect_proposal: "Chem.Mol",
    connect_smiles: "Chem.Mol",
    coords: np.ndarray,
    r_cov: np.ndarray,
    max_matches: int = 10000,
) -> Optional[tuple[int, ...]]:
    """
    this function finds the best match for linking the pyscf atom numbering
    to the SMILES atom numbering.
    the best match is the one with lowest score, with
    score = sum_{SMILES bonds A-B} d_AB / (r_cov,A + r_cov,B)

    arguments:
    - connect_proposal: a proposal for the atom connectivity (i.e. bonds),
                        based on the pyscf coordinates or the bond orders,
                        with pyscf atom numbering
    - connect_smiles: the atom connectivity derived from the SMILES string,
                      with SMILES atom numbering
    - coords: the pyscf coordinates (in Bohr),
              with pyscf atom numbering
    - r_cov: the pyscf covalent radii (in Bohr) of the atoms,
             with pyscf atom numbering
    - max_matches: maximal number of matches that can be considered
    """
    from rdkit import Chem

    Chem.FastFindRings(connect_proposal) # ring information not yet contained in XYZ text
    
    # find all ways to place the SMILES connectivity onto the connectivity proposal
    # e.g. -CH3 groups have six ways (due to rotation and reflection) etc.
    # every match is a tuple: match[i_smiles] = i_pyscf
    matches = connect_proposal.GetSubstructMatches(
        connect_smiles, uniquify=False, maxMatches=max_matches
    )
    if len(matches) == 0:
        return None
    if len(matches) == max_matches:
        logger.warning(
            f"Warning: maximum number of SMILES matches ({max_matches}) reached. "
            "The atom mapping may not be optimal."
        )

    # array with SMILES bonds, identified using atom indices
    bonds = np.array(
        [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in connect_smiles.GetBonds()],
        dtype=int,
    ).reshape(-1, 2)

    # convert the matches from tuples to an array
    m = np.array(matches, dtype=int)  # (n_matches, natm)

    # arrays with begin (ia) and end (ib) atoms of the bonds + mapping from SMILES to pyscf:
    # ia[k, j] = i_pyscf of begin atom of SMILES bond j according to match k
    # ib[k, j] = i_pyscf of end atom of SMILES bond j according to match k
    ia = m[:, bonds[:, 0]]
    ib = m[:, bonds[:, 1]]

    # determine a score:
    # divide bond lengths by the sum of the covalent radii, and sum these ratios
    d = np.linalg.norm(coords[ia] - coords[ib], axis=-1)
    score = np.sum(d / (r_cov[ia] + r_cov[ib]), axis=1)
    
    # return match with lowest score
    return matches[int(np.argmin(score))]
# end def _best_match()

def lewis_bond_mask(
    mol: gto.Mole,
    smiles: Optional[str] = None,
    image_file: str = "lewis_structure.png",
    mbo: Optional[np.ndarray] = None,
    trust_atom_order: bool = False,
    cov_factor: float = 1.8,
    mbo_edge_thresh: float = 0.05,
) -> np.ndarray:
    """
    this function returns a boolean array (ordered according to ap_label) that is True if the
    Lewis structure (generated by RDKit) contains a bond between the atom pair.

    - if no SMILES string is provided:
      the Lewis structure is generated by RDKit based on the provided pyscf coordinates
    - if a SMILES string is provided:
      1. the Lewis structure is generated from the SMILES string
      2. the atom numbering from the SMILES string is mapped to the pyscf atom numbering:
         * if trust_atom_order == True: do not map, assume numberings are the same
         * else: try to map based on atom distances,
                 if this fails: try to map based on bond order values
    
    arguments:
    - mol: the pyscf molecule object
    - smiles: SMILES string for the considered molecule
    - image_file: file name for saving an image of the generated Lewis structure
    - mbo: array of bond orders, with pyscf atom numbering
    - trust_atom_order: whether or not the user guarantees that the pyscf atom ordering
                        and the SMILES atom ordering are the same
    - cov_factor: scaling factor for the sum of covalent radii
                  in the distance-based connectivity proposal
    - mbo_edge_thresh: bond order threshold for an atom pair to be considered an edge
                       in the bond-order-based connectivity proposal
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    atoms_pyscf = _pyscf_to_rdkit(mol) # rdkit object of atoms in pyscf order, no bonds

    if smiles is None:
        # Lewis structure directly from the geometry (xyz2mol)
        logger.warning(
            "Warning: no SMILES string provided. The Lewis structure is determined "
            "from the geometry. Please inspect \"" + image_file + "\"."
        )
        mol_lewis = Chem.Mol(atoms_pyscf)
        rdDetermineBonds.DetermineBonds(mol_lewis, charge=int(mol.charge))
    else:
        # generate a Lewis structure from the provided SMILES string
        params = Chem.SmilesParserParams()
        params.removeHs = False # do not remove hydrogen atoms
        mol_smiles = Chem.MolFromSmiles(smiles, params)
        if mol_smiles is None:
            raise ValueError(f"invalid SMILES string: {smiles}")
        mol_smiles = Chem.AddHs(mol_smiles)
        if mol_smiles.GetNumAtoms() != mol.natm:
            raise ValueError(
                f"SMILES has {mol_smiles.GetNumAtoms()} atoms (incl. H), "
                f"but the molecule has {mol.natm} atoms"
            )

        if trust_atom_order:
            # the user guarantees that pyscf and RDKit atom order are identical
            logger.warning( 
                "Warning: trust_atom_order is set. The Lewis structure is taken from "
                "the SMILES string without checking the atom order. You are fully "
                "responsible for the pyscf atom order being identical to the RDKit "
                "atom order (SMILES atoms in order, followed by the implicit "
                "hydrogens added by RDKit). Please inspect \"" + image_file + "\"."
            )
            mol_lewis = mol_smiles
        else:
            # map the SMILES atom order onto the pyscf atom order

            # generate a connectivity map of only the bonds, from the SMILES
            connect_smiles = _connectivity(mol_smiles)

            # quantities for scoring the matches (Bohr)
            coords = mol.atom_coords()
            r_cov = radii.COVALENT[[a.GetAtomicNum() for a in atoms_pyscf.GetAtoms()]]

            # proposal graph 1: generous distance-based connectivity
            # generate an RDKit mol object with only atoms
            connect_proposal = Chem.Mol(atoms_pyscf)
            # and add single bonds ("edges") between those atom pairs
            # whose interatomic distance is below cov_factor * sum of RDKit's covalent radii
            rdDetermineBonds.DetermineConnectivity(connect_proposal, covFactor=cov_factor)
            # map the SMILES atom numbering onto the pyscf atom numbering
            # by finding connect_smiles as a subgraph of connect_proposal
            match = _best_match(connect_proposal, connect_smiles, coords, r_cov)

            # proposal graph 2 (fallback): MBO-based connectivity
            if match is None and mbo is not None:
                logger.warning(
                    "Warning: SMILES does not match the distance-based connectivity. "
                    "Falling back to MBO-based connectivity."
                )

                # create an editable RDKit mol object
                connect_proposal = Chem.RWMol(atoms_pyscf)

                a_idx, b_idx = np.triu_indices(mol.natm, k=1)
                for a, b in zip(a_idx, b_idx):
                    if mbo[a, b] > mbo_edge_thresh:
                        # add bonds ("edges") between atom pairs with bond order above threshold
                        connect_proposal.AddBond(int(a), int(b), Chem.BondType.SINGLE)

                # map the SMILES atom numbering onto the pyscf atom numbering
                # by finding connect_smiles as a subgraph of connect_proposal
                match = _best_match(connect_proposal, connect_smiles, coords, r_cov)

            if match is None:
                raise ValueError(
                    "the connectivity of the SMILES string could not be mapped "
                    "onto the molecule (neither distance- nor MBO-based)"
                )

            # renumber the SMILES molecule to pyscf order: new_order[i_pyscf] = i_smiles
            new_order = [int(i) for i in np.argsort(match)]
            mol_lewis = Chem.RenumberAtoms(mol_smiles, new_order)
        # end if trust_atom_order
    # end if smiles is None

    # save the produced Lewis structure for the user to inspect
    _save_lewis_image(mol_lewis, image_file)

    # convert RDKit adjacency matrix to an array of booleans
    adj = Chem.GetAdjacencyMatrix(mol_lewis).astype(bool)
    a_idx, b_idx = np.triu_indices(mol.natm, k=1)
    return adj[a_idx, b_idx]
# end def lewis_bond_mask()
