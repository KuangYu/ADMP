#!/usr/bin/env python
import sys
import numpy as np
from itertools import permutations, product
import copy
import jax
import jax.numpy as jnp
from jax import jit, vmap
from admp.settings import jit_condition
from admp.spatial import pbc_shift, v_pbc_shift
from admp.pairwise import distribute_v3


import MDAnalysis as mda

'''
This module works on building graphs based on molecular topology
'''

# def spatial_dr_np(r0, r1, box, box_inv):
#     if box is None:
#         return r1 - r0
#     else:
#         dr = r1 - r0
#         # do the pbc shift thing
#         ds = np.dot(dr, box_inv)
#         ds -= np.floor(ds + 0.5)
#         dr = np.dot(ds, box)
#         return dr

ATYPE_INDEX = {
        'H': 0,
        'C': 1,
        'N': 2,
        'O': 3,
        'S': 4
        }

N_ATYPES = 5

# used to compute equilibrium bond lengths
COVALENT_RADIUS = {
        'H': 0.31,
        'C': 0.76,
        'N': 0.71,
        'O': 0.66,
        'S': 1.05
        }

# scaling parameters for feature calculations
FSCALE_BOND = 10.0
FSCALE_ANGLE = 5.0

MAX_VALENCE = 4
MAX_ANGLES_PER_SITE = MAX_VALENCE * (MAX_VALENCE-1) // 2
MAX_DIHEDS_PER_BOND = (MAX_VALENCE-1) ** 2


class TopGraph:
    '''
    This is the class that describes the topological graph that describes a molecule
    It contains both the topological and the geometrical information of the molecule
    And it is designed to compute the intramolecular energy using the sGNN model.
    '''

    def __init__(self, list_atom_elems, bonds, positions=None, box=None):
        self.list_atom_elems = list_atom_elems
        self.bonds = bonds
        self.n_atoms = len(list_atom_elems)
        self.positions = positions
        self._build_connectivity()
        self._get_valences()
        # debug
        # self.set_internal_coords_indices()
        self.box = box
        if box is not None:
            self.box_inv = jnp.linalg.inv(box)
        else:
            self.box_inv = None
        return


    def set_box(self, box):
        '''
        Set the box information in the class

        Inputs:
            box:
                3 * 3: the box array, pbc vectors arranged in rows
        '''
        self.box = box
        self.box_inv = jnp.linalg.inv(box)
        if hasattr(self, 'subgraphs'):
            self._propagate_attr('box')
            self._propagate_attr('box_inv')
        return


    def set_positions(self, positions, update_subgraph=True):
        '''
        Set positions for the graph/subgraphs
        
        Input:
            positions:
                n * 3, positions matrix
            update_subgraph:
                bool, if we should propogate the positions to the subgraphs or not
        '''
        self.positions = positions
        if update_subgraph:
            self._update_subgraph_positions()
        return


    def _propagate_attr(self, attr):
        '''
        Propogate the attribute of the parent subgraph to each subgraphs
        '''
        # propagate the attribute from the parent graph to the subgraphs
        for ig in range(self.n_subgraphs):
            setattr(self.subgraphs[ig], attr, getattr(self, attr))
        return


    def _build_connectivity(self):
        '''
        Build the connnectivity map in the graph, using the self.bonds information
        '''
        self.connectivity = np.zeros((self.n_atoms, self.n_atoms), dtype=int)
        for i, j in self.bonds:
            self.connectivity[i, j] = 1
            self.connectivity[j, i] = 1
        return 


    def _get_valences(self):
        '''
        Generate the valence number of each atom in the graph
        '''
        if hasattr(self, 'connectivity'):
            self.valences = np.sum(self.connectivity, axis=1)
        else:
            sys.exit('Error in generating valences: build connectivity first!')

    
    def get_all_subgraphs(self, nn, type_center='bond', typify=True, id_chiral=True):
        '''
        Construct all subgraphs from the parent graph, each subgraph contains a central bond/atom
        and its nn'th nearest neighbors. We can choose whether to focus on bonds or focus on atoms

        Inputs:
            nn:
                int, size of the subgraph, 
            type_center:
                str, 'bond' or 'atom', focus on bond or atom?
            typify:
                bool: whether to typify the subgraphs?
            id_chiral:
                bool: while typifying the atoms, whether distinguish chiralities of hydrogens?
                      In particular, in cases like C-ABH2, should we dinstinguish the two hydrogens?

        Output:
            self.subgraphs:
                a list of subgraph objects
        '''
        self.subgraphs = []
        if type_center == 'atom':
            for ia in range(self.n_atoms):
                self.subgraphs.append(TopSubGraph(self, ib, nn, type_center))
        elif type_center == 'bond':
            # build a subgraph around each bond
            for ib, b in enumerate(self.bonds):
                self.subgraphs.append(TopSubGraph(self, ib, nn, type_center))
        self.nn = nn
        self.n_subgraphs = len(self.subgraphs)
        if typify:
            self.typify_all_subgraphs()
        if typify and id_chiral:
            for g in self.subgraphs:
                g._add_chirality_labels()
                # create permutation groups, and canonical orders for atoms
                g.get_canonical_orders_wt_permutation_grps()
        return


    def _update_subgraph_positions(self):
        '''
        pass the positions in the parent graph to subgraphs
        '''
        for g in self.subgraphs:
            g.positions = distribute_v3(self.positions, g.map_sub2parent)
        return


    def get_subgraph(self, i_center, nn, type_center='bond'):
        '''
        Construct a subgraph centered on a certain position

        Input:
            i_center: 
                int, number of the central bond/atoms
            nn:
                int, number of neighbors
            type_center:
                str, bond/atom ?

        Output:
            g:
                the subgraph
        '''
        return TopSubGraph(self, i_center, nn, type_center)


    def typify_atom(self, i, depth=0, excl=None):
        '''
        Typify atom in in the graph
        Use a recursive typification algorithm, similar to MNA in openbabel

        Input:
            i:
                int, the index of the atom to typify
            depth:
                int, depth of recursion
            excl:
                the exclusion atom idex, only used for recursion
        '''
        if depth == 0:
            return self.list_atom_elems[i]
        else: # recursive execution
            atype = self.list_atom_elems[i]
            atype_nbs = []
            for j in np.where(self.connectivity[i] == 1)[0]:
                if j != excl:
                    atype_nbs.append(self.typify_atom(j, depth=depth-1, excl=i))
            atype_nbs.sort()
            if len(atype_nbs) == 0:
                return atype
            else:
                atype = atype + '-(' + ','.join(atype_nbs) + ')'
                return atype


    def typify_all_atoms(self, depth=0):
        '''
        Typify all atoms in graph
        '''
        self.atom_types = []
        for i in range(self.n_atoms):
            self.atom_types.append(self.typify_atom(i, depth=depth))
        self.atom_types = np.array(self.atom_types)
        return


    def typify_subgraph(self, i):
        '''
        Do typification to subgraph i
        the depth is set to be 2*nn + 4, that is the largest possible size of subgraphs
        '''
        self.subgraphs[i].typify_all_atoms(depth=(2*self.nn+4))
        return


    def typify_all_subgraphs(self):
        '''
        Do typification to all subgraphs
        '''
        for i_subgraph in range(self.n_subgraphs):
            self.typify_subgraph(i_subgraph)
        return


    def _add_chirality_labels(self):
        '''
        This subroutine add labels to distinguish hydrogens in ABCH2
        It uses the position info to identify the chirality of the H
        '''
        for i in range(self.n_atoms):
            neighbors = np.where(self.connectivity[i] == 1)[0]
            if len(neighbors) != 4:
                continue
            labels = self.atom_types[neighbors]
            flags = np.array([labels==labels[i] for i in range(4)])
            flags1 = flags.sum(axis=1)
            if np.sum(flags) == 6: # C-ABH2
                filter_H = (flags.sum(axis=1)==2)
                j, k = neighbors[np.where(filter_H)[0]]
                l, m = neighbors[np.where(np.logical_not(filter_H))[0]]
                ti, tj, tk, tl, tm = self.atom_types[[i, j, k, l, m]]
                # swap l and m, such that tl < tm
                if tl > tm:
                    (l, m) = (m, l)
                    tl, tm = np.array(self.atom_types)[[l, m]]
                ri, rj, rk, rl, rm = self.positions[jnp.array([i, j, k, l, m])]
                rij = pbc_shift(rj - ri, self.box, self.box_inv)
                rkl = pbc_shift(rl - rk, self.box, self.box_inv)
                rkm = pbc_shift(rm - rk, self.box, self.box_inv)
                if jnp.dot(rij, jnp.cross(rkl, rkm)) > 0:
                    self.atom_types[j] += 'R'
                    self.atom_types[k] += 'L'
                else:
                    self.atom_types[j] += 'L'
                    self.atom_types[k] += 'R'
        return


    def set_internal_coords_indices(self):
        '''
        This method go over the graph and search for all bonds, angles, diheds
        It records the atom indices for all ICs, and also the equilibrium bond lengths and angles
        It sets the following attributes in the graph:
        bonds, a0, angles, cos_a0, diheds
        n_bonds, n_angles, n_diheds
        '''
        # bonds
        self.bonds = np.array(self.bonds)
        # equilibrium bond lengths
        a0 = self.bonds[:, 0]
        a1 = self.bonds[:, 1]
        at0 = self.list_atom_elems[a0]
        at1 = self.list_atom_elems[a1]
        r0 = jnp.array([COVALENT_RADIUS[e0] for e0 in at0])
        r1 = jnp.array([COVALENT_RADIUS[e1] for e1 in at1])
        self.b0 = r0 + r1
        self.n_bonds = len(self.bonds)

        #angles
        angles = []
        for i in range(self.n_atoms):
            neighbors = np.where(self.connectivity[i] == 1)[0]
            for jj, j in enumerate(neighbors):
                for kk, k in enumerate(neighbors[jj+1:]):
                    angles.append([j, i, k])
        self.angles = np.array(angles)
        def get_a0(indices_angles):
            a0 = np.zeros(len(indices_angles))
            for ia, (j, i, k) in enumerate(indices_angles):
                if i >=0 and j >= 0 and k >= 0:
                    valence = self.valences[i]
                    if valence == 2 and self.list_atom_elems[i] == 'O' or self.list_atom_elems[i] == 'S':
                        cos_a0 = np.cos(104.45/180*np.pi)
                    elif valence == 2 and self.list_atom_elems[i] == 'N':
                        cos_a0 = np.cos(120./180*np.pi)
                    elif valence == 2:
                        cos_a0 = np.cos(np.pi)
                    elif valence == 3 and self.list_atom_elems[i] == 'N':
                        cos_a0 = np.cos(107./180*np.pi)
                    elif valence == 3:
                        cos_a0 = np.cos(120.00/180*np.pi)
                    elif valence == 4:
                        cos_a0 = np.cos(109.45/180*np.pi) # 109.5 degree
                    a0[ia] = cos_a0
            return a0
        self.cos_a0 = jnp.array(get_a0(self.angles))
        self.n_angles = len(self.angles)
        # diheds
        diheds = []
        for ib in range(len(self.bonds)):
            j, k = self.bonds[ib]
            ilist = np.where(self.connectivity[j] == 1)[0]
            llist = np.where(self.connectivity[k] == 1)[0]
            for i in ilist:
                if i == k:
                    continue
                for l in llist:
                    if l == j:
                        continue
                    diheds.append([i, j, k, l])
        self.diheds = jnp.array(diheds)
        self.n_diheds = len(self.diheds)
        return


    @jit_condition(static_argnums=())
    def calc_internal_coords_features(self, positions, box):
        '''
        Calculate the feature value of all ICs in the subgraph
        This function meant to be exposed to external use, with jit and grad etc.
        It relies on the following variables in Graph:
        self.bonds, self.angles, self.diheds
        self.a0, self.cos_b0
        All these variables should be "static" throughout NVE/NVT/NPT simulations
        '''
       
        box_inv = jnp.linalg.inv(box)
        @jit_condition(static_argnums=())
        @partial(vmap, in_axes=(0, None, 0), out_axes=(0))
        def _calc_bond_features(idx, pos, b0):
            pos0 = pos[idx[0]]
            pos1 = pos[idx[1]]
            dr = pbc_shift(pos1 - pos0, box, box_inv)
            blength = jnp.linalg.norm(dr)
            return (blength - b0) * FSCALE_BOND

        
        @jit_condition(static_argnums=())
        @partial(vmap, in_axes=(0, None, 0), out_axes=(0))
        def _calc_angle_features(idx, pos, cos_a0):
            rj = pos[idx[0]]
            ri = pos[idx[1]]
            rk = pos[idx[2]]
            r_ij = pbc_shift(rj - ri, box, box_inv)
            r_ik = pbc_shift(rk - ri, box, box_inv)
            n_ij = jnp.linalg.norm(r_ij)
            n_ik = jnp.linalg.norm(r_ik)
            cos_a = jnp.dot(r_ij, r_ik) / n_ij / n_ik
            return (cos_a - cos_a0) * FSCALE_ANGLE

        @jit_condition(static_argnums=())
        @partial(vmap, in_axes=(0, None), out_axes=(0))
        def _calc_dihed_features(idx, pos):
            ri = pos[idx[0]]
            rj = pos[idx[0]]
            rk = pos[idx[0]]
            rl = pos[idx[0]]
            r_jk = pbc_shift(rk - rj, box, box_inv)
            r_ji = pbc_shift(ri - rj, box, box_inv)
            r_kl = pbc_shift(rl - rk, box, box_inv)
            r_kj = -r_jk
            n1 = jnp.cross(r_jk, r_ji)
            n2 = jnp.cross(r_kl, r_kj)
            norm_n1 = jnp.linalg.norm(n1)
            norm_n2 = jnp.linalg.norm(n2)
            return jnp.dot(n1, n2) / norm_n1 / norm_n2

        fb = _calc_bond_features(self.bonds, positions, self.b0)
        fa = _calc_angle_features(self.angles, positions, self.cos_a0)
        fd = _calc_dihed_features(self.diheds, positions)

        return fb, fa, fd



    def prepare_subgraph_feature_calc(self):
        for g in self.subgraphs:
            g.prepare_graph_feature_calc()
        return

    # def calc_subgraph_features(self):
    #     self.calc_internal_coords_features()
    #     for g in self.subgraphs

            # flag = False
            # for p in permutations([0, 1, 2, 3]):
            #     j, k, l, m = neighbors[list(p)]
            #     labels = np.array(self.atom_types)[[j, k, l, m]]
            #     # find a chiral label case
            #     if labels[0] == labels[1] and labels[0] != labels[2] and labels[0] != labels[3] and labels[2] != labels[3]:
            #         flag = True
            #         break

                    
def from_pdb(pdb):
    '''
    This is the old version using mda
    '''
    u = mda.Universe(pdb)
    list_atom_elems = np.array(u.atoms.types)
    bonds = []
    for bond in u.bonds:
        bonds.append(np.sort(bond.indices))
    bonds = np.array(bonds)
    positions = jnp.array(u.atoms.positions)
    if np.sum(np.abs(u.dimensions)) < 1e-8:  # no box information
        box = None
    else:
        box = jnp.array(mda.lib.mdamath.triclinic_vectors(u.dimensions))
    return TopGraph(list_atom_elems, bonds, positions=positions, box=box)


if __name__ == '__main__':
    graph_mol = from_pdb('peg4.pdb')
    graph_mol.set_internal_coords_indices()
    nn = 1
