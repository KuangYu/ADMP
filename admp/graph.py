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
        You can also choose wheter to typify the atoms in the subgraphs and build canonical orders

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
        self.atom_types = np.array(self.atom_types, dtype="object")
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


    def _add_chirality_labels(self, verbose=False):
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
                    tl, tm = np.array(self.atom_types, dtype="object")[[l, m]]
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
        '''
        Preparing the feature calculation.
        Specifically, find out the indices mapping between feature elements and ICs
        '''
        for g in self.subgraphs:
            g.prepare_graph_feature_calc()
        return


    def calc_subgraph_features(self):
        # TODO
        pass


    def write_xyz(self, file=None):
        '''
        Write the structure of the subgraph in xyz format
        '''
        if file is None:
            ofile = sys.stdout
        else:
            ofile = open(file, 'w')
        print(self.n_atoms, file=ofile)
        print('generated by the write_xyz function in TopGraph', file=ofile)
        for i in range(self.n_atoms):
            r = self.positions[i]
            print('%3s%15.8f%15.8f%15.8f'%(self.list_atom_elems[i], r[0], r[1], r[2]), file=ofile)
        if file is not None:
            ofile.close()


class TopSubGraph(TopGraph):

    def __init__(self, graph, i_center, nn, type_center='bond'):
        '''
        Find a subgraph within the graph, centered on a certain bond/atom
        The size of the subgraph is determined by nn (# of neighbour searches around the center)
        i_center defines the center, could be a bond, could be an atom
        '''
        self.list_atom_elems = []
        self.bonds = []
        self.positions = []
        self.valences = []
        self.map_sub2parent = [] # this maps the index in the subgraph to the index in the parent graph
        self.map_parent2sub = {}
        self.parent = graph
        self.box = graph.box
        self.box_inv = graph.box_inv
        self.nn = nn
        n_atoms = 0
        if type_center == 'atom':
            self.map_sub2parent.append(i_center)
            self.map_parent2sub[i_center] = n_atoms
            n_atoms += 1
            self.list_atom_elems.append(graph.list_atom_elems[i_center])
            self.valences.append(graph.valences[i_center])
        elif type_center == 'bond':
            b0 = graph.bonds[i_center]
            for i in b0:
                self.map_sub2parent.append(i)
                self.map_parent2sub[i] = n_atoms
                n_atoms += 1
                self.list_atom_elems.append(graph.list_atom_elems[i])
                self.valences.append(graph.valences[i])
            # the first bond of the subgraph is always (0, 1), the central bond
            self.bonds.append(np.array([0, 1]))
        self.n_atoms = n_atoms

        for n in range(nn+1):
            self.add_neighbors()
        self._build_connectivity()

        self.map_sub2parent = np.array(self.map_sub2parent)
        self.positions = graph.positions[self.map_sub2parent]

        return


    # search one more layer of neighbours
    def add_neighbors(self):
        atoms_in_subgraph = list(self.map_parent2sub.keys())
        n_atoms = self.n_atoms
        for b in self.parent.bonds:
            flags = np.array([not (i in atoms_in_subgraph) for i in b])
            if np.sum(flags) == 1:
                i_old = np.array(b)[[not f for f in flags]][0]
                i_new = np.array(b)[flags][0]
                self.list_atom_elems.append(self.parent.list_atom_elems[i_new])
                self.positions.append(self.parent.positions[i_new])
                self.valences.append(self.parent.valences[i_new])
                self.map_sub2parent.append(i_new)
                self.map_parent2sub[i_new] = n_atoms
                bond = np.array([n_atoms, self.map_parent2sub[i_old]])
                self.bonds.append(np.sort(bond))
                n_atoms += 1
        self.n_atoms = n_atoms
        return 


    def get_canonical_orders_wt_permutation_grps(self):
        '''
        This function sets up all the canonical orders for the atoms, based on existing 
        atom typification (atom_types) information and the connection topology.
        Specifically, it sets the following variables in the subgraph:

        g.canonical_orders
            All the orders that are symmetrically equivalent and nondistinguishable
        g.maps_canonical_orders
            The reverse mapping of the canonical orders (i.e., maps from atom indices to order)
        g.n_permutation
            Number of canonical orders
        '''
        # 'orders' is a queue that stores all the sequences
        if self.atom_types[0] == self.atom_types[1]:
            orders = [np.array([0, 1]), np.array([1, 0])]
        else:
            t0 = self.atom_types[0]
            t1 = self.atom_types[1]
            if t0 < t1:
                orders = [np.array([0, 1])]
            else:
                orders = [np.array([1, 0])]
        # generate permutations of indices according to the atypes
        # atoms are sorted according to their types, and the atoms
        # with the same type will be permuted
        def permute_using_atypes(indices, atypes):
            set_atypes = list(set(atypes))
            set_atypes.sort()
            sym_classes = {}
            permutation_grps = []
            for t in set_atypes:
                sym_classes[t] = np.where(atypes==t)[0]
                permutation_grps.append(indices[sym_classes[t]])
            # permute, create index orders
            indices_permutations = []
            dict_indices_range = []
            seg_permutations = []
            for pseg in permutation_grps:
                seg_permutations.append(list(permutations(pseg)))
            pfull = []
            for p in product(*seg_permutations):
                pfull.append(np.concatenate(p))
            return np.array(pfull)
        # go through the queue, and grow each sequence (order) by one
        # more layer of atom
        def extend_orders(orders):
            n_order = len(orders)
            # do a loop over all existing sequences
            for i_order in range(n_order):
                order = orders.pop(0) 
                seg_permutations = []
                for i in order:
                    js = np.where(self.connectivity[i])[0]
                    # remove those already in sequence
                    js = js[[not (j in order) for j in js]]
                    if len(js) == 0:
                        continue
                    # now permute js
                    atypes = np.array(self.atom_types)[js]
                    new_orders = permute_using_atypes(js, atypes)
                    seg_permutations.append(new_orders)
                for p in product(*seg_permutations):
                    orders.append(np.concatenate((order, np.concatenate(p))))
            return orders

        # starting from the two atoms in the central bond, grow two layers
        # should have all the atoms included
        for i in range(self.nn+1):
            orders = extend_orders(orders)
        canonical_orders = np.array(orders)
        maps_canonical_orders = []
        for order in canonical_orders:
            map_order = np.zeros(self.n_atoms, dtype=int)
            for ii, i in enumerate(order):
                map_order[i] = ii
            maps_canonical_orders.append(map_order)
        maps_canonical_orders = np.array(maps_canonical_orders)
        # all the possible canonical orders, and the corresponding index mapping 
        self.canonical_orders = canonical_orders
        self.maps_canonical_orders = maps_canonical_orders
        self.n_permutations = len(canonical_orders)
        return

    
    # the feature that represents the typification info
    def prepare_bond_feature_atypes(self, bond, map_order):
        '''
        Get feature elements that label the atom types
        For each atom, a vector is specified to mark its element
        [1 0 0 0 0] is H
        [0 1 0 0 0] is C
        [0 0 1 0 0] is N
        etc.
        These vectors are then catenated according to the given canonical order
        '''
        indices_atoms_center = np.array(bond)
        indices_atoms_center = sort_by_order(indices_atoms_center, map_order)
        # indices in the subgraph
        i, j = indices_atoms_center
        # elements
        elem_i = self.list_atom_elems[i]
        elem_j = self.list_atom_elems[j]
        fi = np.zeros(N_ATYPES)
        fj = np.zeros(N_ATYPES)
        fi[ATYPE_INDEX[elem_i]] = 1
        fj[ATYPE_INDEX[elem_j]] = 1
        # neighbour atoms
        indices_n0 = np.array(np.where(self.connectivity[i]==1)[0])
        indices_n1 = np.array(np.where(self.connectivity[j]==1)[0])
        indices_n0 = indices_n0[indices_n0!=j]
        indices_n1 = indices_n1[indices_n1!=i]
        # sort with canonical order
        indices_n0 = sort_by_order(indices_n0, map_order)
        indices_n1 = sort_by_order(indices_n1, map_order)
        nn0 = len(indices_n0)
        nn1 = len(indices_n1)
        # features of the neighbour atom types
        f_n0 = np.zeros(N_ATYPES*(MAX_VALENCE-1))
        f_n1 = np.zeros(N_ATYPES*(MAX_VALENCE-1))
        for ii, i in enumerate(indices_n0):
            tmp = np.zeros(N_ATYPES)
            elem = self.list_atom_elems[i]
            tmp[ATYPE_INDEX[elem]] = 1
            f_n0[ii*N_ATYPES:ii*N_ATYPES+N_ATYPES] = tmp
        for ii, i in enumerate(indices_n1):
            tmp = np.zeros(N_ATYPES)
            elem = self.list_atom_elems[i]
            tmp[ATYPE_INDEX[elem]] = 1
            f_n1[ii*N_ATYPES:ii*N_ATYPES+N_ATYPES] = tmp
        return jnp.array(np.concatenate((fi, fj, f_n0, f_n1)))

    
    def prepare_bond_feature_calc_indices(self, bond, map_order):
        '''
        Given a bond, and a particular order of the atoms in the graph, prepare its
        geometric feature calculations.
        The geometric features of a bond will be composed by:
        1. It's own lengths
        2. The lengths of all it's neighbor bonds
        3. All angles that share atoms with the bond
        4. All diheds that are centered on the bond

        Correspondingly, we prepare the indices (in parent graph) of the corresponding ICs:
        indices['bond']: indices for all relevant bonds
        indices['angles[12]']: indices for all relevant angles
        indices['diheds']: indices for all relevant diheds

        All IC indices will be sorted according to the given atomic order.
        '''
        indices = {}
        G = self.parent
        indices_atoms_center = np.array(bond)
        indices_atoms_center = sort_by_order(indices_atoms_center, map_order)
        i, j = indices_atoms_center
        indices_n0 = np.array(np.where(self.connectivity[i]==1)[0])
        indices_n1 = np.array(np.where(self.connectivity[j]==1)[0])
        indices_n0 = indices_n0[indices_n0!=j]
        indices_n1 = indices_n1[indices_n1!=i]
        indices_n0 = sort_by_order(indices_n0, map_order)
        indices_n1 = sort_by_order(indices_n1, map_order)
        nn0 = len(indices_n0)
        nn1 = len(indices_n1)
        # padding neighbours
        indices_atoms_n0 = -np.ones(MAX_VALENCE-1, dtype=int)
        indices_atoms_n1 = -np.ones(MAX_VALENCE-1, dtype=int)
        indices_atoms_n0[:nn0] = indices_n0
        indices_atoms_n1[:nn1] = indices_n1

        # relevant bonds
        indices_bonds = []
        indices_bonds.append(indices_atoms_center)
        for i in indices_atoms_n0:
            indices_bonds.append([indices_atoms_center[0], i])
        for j in indices_atoms_n1:
            indices_bonds.append([indices_atoms_center[1], j])
        indices_bonds = np.array(indices_bonds)
        # convert to indices in parent graph
        indices['bonds'] = []
        for b in indices_bonds:
            p = np.array([self.map_sub2parent[i] for i in b])
            match = np.where(np.all(G.bonds == p, axis=1) + np.all(G.bonds == p[::-1], axis=1))[0]
            if len(match) == 0:
                indices['bonds'].append(-1)
            else:
                indices['bonds'].append(match[0])
        indices['bonds'] = np.array(indices['bonds'], dtype=int)

        # relevant angles
        indices_angles_0 = []
        set_0 = np.array([indices_atoms_center[1]] + list(indices_atoms_n0))
        for ii, i in enumerate(set_0):
            for jj in range(ii+1, len(set_0)):
                j = set_0[jj]
                angle = [i, indices_atoms_center[0], j]
                indices_angles_0.append(angle)
        indices_angles_0 = np.array(indices_angles_0, dtype=int)
        indices_angles_1 = []
        set_1 = np.array([indices_atoms_center[0]] + list(indices_atoms_n1))
        for ii, i in enumerate(set_1):
            for jj in range(ii+1, len(set_1)):
                j = set_1[jj]
                angle = [i, indices_atoms_center[1], j]
                indices_angles_1.append(angle)
        indices_angles_1 = np.array(indices_angles_1, dtype=int)
        # convert to indices in parent graph
        indices['angles0'] = []
        indices['angles1'] = []
        for a in indices_angles_0:
            p = np.array([self.map_sub2parent[i] for i in a])
            match = np.where(np.all(G.angles == p, axis=1) + np.all(G.angles == p[::-1], axis=1))[0]
            if len(match) == 0:
                indices['angles0'].append(-1)
            else:
                indices['angles0'].append(match[0])
        for a in indices_angles_1:
            p = np.array([self.map_sub2parent[i] for i in a])
            match = np.where(np.all(G.angles == p, axis=1) + np.all(G.angles == p[::-1], axis=1))[0]
            if len(match) == 0:
                indices['angles1'].append(-1)
            else:
                indices['angles1'].append(match[0])
        indices['angles0'] = np.array(indices['angles0'], dtype=int)
        indices['angles1'] = np.array(indices['angles1'], dtype=int)

        # relevant dihedrals
        indices_diheds = []
        for i in indices_atoms_n0:
            for j in indices_atoms_n1:
                indices_diheds.append([i, indices_atoms_center[0], indices_atoms_center[1], j])
        indices_diheds = np.array(indices_diheds)
        indices['diheds'] = []
        for d in indices_diheds:
            p = np.array([self.map_sub2parent[i] for i in d])
            match = np.where(np.all(G.diheds == p, axis=1) + np.all(G.diheds == p[::-1], axis=1))[0]
            if len(match) == 0:
                indices['diheds'].append(-1)
            else:
                indices['diheds'].append(match[0])
        indices['diheds'] = np.array(indices['diheds'], dtype=int)

        # number of features to describe a bond
        self.n_bond_features = np.sum([len(indices[k]) for k in indices.keys()]) + 8*N_ATYPES
        return indices
        

    def prepare_graph_feature_calc(self):
        # assume the first bond is always the central bond
        center_bond = self.bonds[0] # should always be (0, 1)
        if self.nn = 0:
            self.weights

        
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


def sort_by_order(ilist, map_order):
    '''
    Sort the list according to the given canonical order

    Input:
        map_order:
            int list: maps from atom indices to its canonical order
        ilist:
            int array: atom indices that needs to be sorted

    Output:
        ilist_new:
            int array: atom indices that are sorted
    '''
    return np.array(ilist)[np.argsort([map_order[i] for i in ilist])]

                    
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
    G = from_pdb('peg4.pdb')
    nn = 0
    G.get_all_subgraphs(nn, typify=True)
    g = G.subgraphs[4]
    g.write_xyz()
    map_order = g.maps_canonical_orders[0]
    print(g.prepare_bond_feature_calc_indices(g.bonds[0], map_order))
