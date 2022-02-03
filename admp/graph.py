#!/usr/bin/env python
import sys
import numpy as np
from itertools import permutations, product
import copy
import jax
import jax.numpy as jnp
from jax import jit, vmap
from admp.spatial import pbc_shift, v_pbc_shift


'''
This module works on building graphs based on molecular topology
'''


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
        self.valences = self._get_valences()
        self.set_internal_coords_indices()
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
        return valences

    
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
            self.update_subgraph_positions()
        return


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
            sys.exit('Error: type center atom is not implemented yet!')
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
            g.add_H_chirality_labels()
            # create permutation groups, and canonical orders for atoms
            g.get_canonical_orders_wt_permutation_grps()
        return


    def update_subgraph_positions(self):
        '''
        pass the positions in the parent graph to subgraphs
        '''
        for g in self.subgraphs:
            indices = 
