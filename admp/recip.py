#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad, value_and_grad
import admp.settings
from admp.settings import DO_JIT, jit_condition
from admp.pme import DIELECTRIC

# for debug
# from admp.parser import *
# from admp.multipole import *
# from admp.spatial import *
# from admp.pme import *
# from jax.config import config
# config.update("jax_enable_x64", True)

sqrt_pi = 1.7724538509055159

def generate_pme_recip(Ck_fn, kappa, gamma, pme_order, K1, K2, K3, lmax):

    # Currently only supports pme_order=6
    # Because only the 6-th order spline function is hard implemented
    pme_order = 6
    # global variables for the reciprocal module, all related to pme_order
    bspline_range = jnp.arange(-pme_order//2, pme_order//2)
    n_mesh = pme_order**3
    shifts = jnp.array(jnp.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))
   
    def pme_recip(positions, box, Q):
        '''
        The generated pme_recip space calculator
        kappa, pme_order, K1, K2, K3, and lmax are passed and fixed when the calculator is generated
        '''
    
        def get_recip_vectors(N, box):
            """
            Computes reciprocal lattice vectors of the grid
            
            Input:
                N:
                    (3,)-shaped array, (K1, K2, K3)
                box:
                    3 x 3 matrix, box parallelepiped vectors arranged in TODO rows or columns?
                    
            Output: 
                Nj_Aji_star:
                    3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                    (lattice vectors arranged in rows)
            """
            Nj_Aji_star = (N.reshape((1, 3)) * jnp.linalg.inv(box)).T
            return Nj_Aji_star
        
        
        def u_reference(R_a, Nj_Aji_star):
            """
            Each atom is meshed to dispersion_ORDER**3 points on the m-meshgrid. 
            This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates,
            and the corresponding values of xyz fractional displacements from real coordinate to the reference point. 
            
            Inputs:
                R_a:
                    N_a * 3 matrix containing positions of sites
                Nj_Aji_star:
                    3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                    (lattice vectors arranged in rows)
                    
            Outputs:
                m_u0: 
                    N_a * 3 matrix, positions of the reference points of R_a on the m-meshgrid
                u0: 
                    N_a * 3 matrix, (R_a - R_m)*a_star values
            """
            R_in_m_basis =  jnp.einsum("ij,kj->ki", Nj_Aji_star, R_a)
            m_u0 = jnp.ceil(R_in_m_basis).astype(int)
            u0 = (m_u0 - R_in_m_basis) + pme_order/2
            return m_u0, u0
        
        def bspline(u, order=pme_order):
            """
            Computes the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(u, 
                                [jnp.logical_and(u>=0, u<1.), 
                                jnp.logical_and(u>=1, u<2.), 
                                jnp.logical_and(u>=2, u<3.), 
                                jnp.logical_and(u>=3, u<4.), 
                                jnp.logical_and(u>=4, u<5.), 
                                jnp.logical_and(u>=5, u<6.)],
                                [lambda u: u**5/120,
                                lambda u: u**5/120 - (u - 1)**5/20,
                                lambda u: u**5/120 + (u - 2)**5/8 - (u - 1)**5/20,
                                lambda u: u**5/120 - (u - 3)**5/6 + (u - 2)**5/8 - (u - 1)**5/20,
                                lambda u: u**5/24 - u**4 + 19*u**3/2 - 89*u**2/2 + 409*u/4 - 1829/20,
                                lambda u: -u**5/120 + u**4/4 - 3*u**3 + 18*u**2 - 54*u + 324/5] )
        
        
        def bspline_prime(u, order=pme_order):
            """
            Computes first derivative of the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(u, 
                                [jnp.logical_and(u>=0., u<1.), 
                                jnp.logical_and(u>=1., u<2.), 
                                jnp.logical_and(u>=2., u<3.), 
                            jnp.logical_and(u>=3., u<4.), 
                            jnp.logical_and(u>=4., u<5.), 
                            jnp.logical_and(u>=5., u<6.)],
                            [lambda u: u**4/24,
                            lambda u: u**4/24 - (u - 1)**4/4,
                            lambda u: u**4/24 + 5*(u - 2)**4/8 - (u - 1)**4/4,
                            lambda u: -5*u**4/12 + 6*u**3 - 63*u**2/2 + 71*u - 231/4,
                            lambda u: 5*u**4/24 - 4*u**3 + 57*u**2/2 - 89*u + 409/4,
                            lambda u: -u**4/24 + u**3 - 9*u**2 + 36*u - 54] )


        def bspline_prime2(u, order=pme_order):
            """
            Computes second derivate of the cardinal B-spline function
            """
            if order == 6:
                return jnp.piecewise(u, 
                                [jnp.logical_and(u>=0., u<1.), 
                                jnp.logical_and(u>=1., u<2.), 
                                jnp.logical_and(u>=2., u<3.), 
                                jnp.logical_and(u>=3., u<4.), 
                                jnp.logical_and(u>=4., u<5.), 
                                jnp.logical_and(u>=5., u<6.)],
                                [lambda u: u**3/6,
                                lambda u: u**3/6 - (u - 1)**3,
                                lambda u: 5*u**3/3 - 12*u**2 + 27*u - 19,
                                lambda u: -5*u**3/3 + 18*u**2 - 63*u + 71,
                                lambda u: 5*u**3/6 - 12*u**2 + 57*u - 89,
                                lambda u: -u**3/6 + 3*u**2 - 18*u + 36,] )
        
        
        def theta_eval(u, M_u):
            """
            Evaluates the value of theta given 3D u values at ... points 
            
            Input:
                u:
                    ... x 3 matrix
        
            Output:
                theta:
                    ... matrix
            """
            theta = jnp.prod(M_u, axis = -1)
            return theta
        
        
        def thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u):
            """
            First derivative of theta with respect to x,y,z directions
            
            Input:
                u
                Nj_Aji_star:
                    reciprocal lattice vectors
            
            Output:
                N_a * 3 matrix
            """
        
            div = jnp.array([
                Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
                Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
                Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            ]).T
            
            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("ij,kj->ki", -Nj_Aji_star, div)
        

        def theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
            """
            compute the 3 x 3 second derivatives of theta with respect to xyz
            
            Input:
                u
                Nj_Aji_star
            
            Output:
                N_A * 3 * 3
            """

            div_00 = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
            div_11 = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
            div_22 = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]
            
            div_01 = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
            div_02 = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
            div_12 = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]

            div_10 = div_01
            div_20 = div_02
            div_21 = div_12
            
            div = jnp.array([
                [div_00, div_01, div_02],
                [div_10, div_11, div_12],
                [div_20, div_21, div_22],
            ]).swapaxes(0, 2)
            
            # Notice that u = m_u0 - R_in_m_basis + 6/2
            # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
            return jnp.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)


        def sph_harmonics_GO(u0, Nj_Aji_star):
            '''
            Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
            00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
            Currently supports lmax <= 2
        
            Inputs:
                u0: 
                    a N_a * 3 matrix containing all positions
                Nj_Aji_star:
                    reciprocal lattice vectors in the m-grid
                lmax:
                    int: max L
        
            Output: 
                harmonics: 
                    a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
                    evaluated at 6*6*6 integer points about reference points m_u0 
            '''
            
            n_harm = (lmax + 1)**2
        
            N_a = u0.shape[0]
            # mesh points around each site
            u = (u0[:, jnp.newaxis, :] + shifts).reshape((N_a*n_mesh, 3)) 
        
            M_u = bspline(u)
            theta = theta_eval(u, M_u)
            if lmax == 0:
                return theta.reshape(N_a, n_mesh, n_harm)
            
            # dipole
            Mprime_u = bspline_prime(u)
            thetaprime = thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u)
            harmonics_1 = jnp.stack(
                [theta,
                thetaprime[:, 2],
                thetaprime[:, 0],
                thetaprime[:, 1]],
                axis = -1
            )
            
            if lmax == 1:
                return harmonics_1.reshape(N_a, n_mesh, n_harm)
        
            # quadrapole
            M2prime_u = bspline_prime2(u)
            theta2prime = theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u)
            rt3 = jnp.sqrt(3)
            harmonics_2 = jnp.hstack(
                [harmonics_1,
                jnp.stack([(3*theta2prime[:,2,2] - jnp.trace(theta2prime, axis1=1, axis2=2)) / 2,
                rt3 * theta2prime[:, 0, 2],
                rt3 * theta2prime[:, 1, 2],
                rt3/2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1]),
                rt3 * theta2prime[:, 0, 1]], axis = 1)]
            )
            if lmax == 2:
                return harmonics_2.reshape(N_a, n_mesh, n_harm)
            else:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')
        
        
        def Q_m_peratom(Q, sph_harms):
            """
            Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983
            
            Inputs:
                Q: 
                    N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
                sph_harms:
                    N_a, 216, (l+1)**2
                lmax:
                    int: maximal L
            
            Output:
                Q_m_pera:
                    N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
            """
            
            N_a = sph_harms.shape[0]
            
            if lmax > 2:
                raise NotImplementedError('l > 2 (beyond quadrupole) not supported')
            
            Q_dbf = Q[:, 0:1]

            if lmax >= 1:
                Q_dbf = jnp.hstack([Q_dbf, Q[:,1:4]])
            if lmax >= 2:
                Q_dbf = jnp.hstack([Q_dbf, Q[:,4:9]/3])
           
            Q_m_pera = jnp.sum(Q_dbf[:,jnp.newaxis,:]* sph_harms, axis=2)
                                                                                                 
            assert Q_m_pera.shape == (N_a, n_mesh)
            return Q_m_pera
        
        
        def Q_mesh_on_m(Q_mesh_pera, m_u0, N):
            """
            Reduce the local Q_m_peratom into the global mesh
            
            Input:
                Q_mesh_pera, m_u0, N
                
            Output:
                Q_mesh: 
                    Nx * Ny * Nz matrix
            """
            indices_arr = jnp.mod(m_u0[:,np.newaxis,:]+shifts, N[np.newaxis, np.newaxis, :])
            ### jax trick implementation without using for loop
            ### NOTICE: this implementation does not work with numpy!
            Q_mesh = jnp.zeros((N[0], N[1], N[2]))
            Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)
            return Q_mesh


        def setup_kpts_integer(N):
            """
            Outputs:
                kpts_int:
                    n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
            """
            N_half = N.reshape(3)
            kx, ky, kz = [jnp.roll(jnp.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2 ), - (N_half[i] - 1) // 2) for i in range(3)]
            kpts_int = jnp.hstack([ki.flatten()[:,jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
            return kpts_int 


        def setup_kpts(box, kpts_int):
            '''
            This function sets up the k-points used for reciprocal space calculations
            
            Input:
                box:
                    3 * 3, three axis arranged in rows
                kpts_int:
                    n_k * 3 matrix

            Output:
                kpts:
                    4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
            '''
            # in this array, a*, b*, c* (without 2*pi) are arranged in column
            box_inv = jnp.linalg.inv(box)
            # K * 3, coordinate in reciprocal space
            kpts = 2 * jnp.pi * kpts_int.dot(box_inv)
            ksq = jnp.sum(kpts**2, axis=1)
            # 4 * K
            kpts = jnp.hstack((kpts, ksq[:, jnp.newaxis])).T
            return kpts


        def spread_Q(positions, box, Q):
            '''
            This is the high level wrapper function, in charge of spreading the charges/multipoles on grid

            Input:
                positions:
                    Na * 3: positions of each site
                box: 
                    3 * 3: box
                Q:
                    Na * (lmax+1)**2: the multipole of each site in global frame

            Output:
                Q_mesh:
                    K1 * K2 * K3: the meshed multipoles
                
            '''
            Nj_Aji_star = get_recip_vectors(N, box)
            # For each atom, find the reference mesh point, and u position of the site
            m_u0, u0 = u_reference(positions, Nj_Aji_star)
            # find out the STGO values of each grid point
            sph_harms = sph_harmonics_GO(u0, Nj_Aji_star)
            # find out the local meshed values for each site
            Q_mesh_pera = Q_m_peratom(Q, sph_harms)
            return Q_mesh_on_m(Q_mesh_pera, m_u0, N)

        # spread Q
        N = np.array([K1, K2, K3])
        Q_mesh = spread_Q(positions, box, Q)
        N = N.reshape(1, 1, 3)
        kpts_int = setup_kpts_integer(N)
        kpts = setup_kpts(box, kpts_int)
        m = jnp.linspace(-pme_order//2+1, pme_order//2-1, pme_order-1).reshape(pme_order-1, 1, 1)
        # m = jnp.linspace(-2,2,5).reshape(5, 1, 1)
        theta_k = jnp.prod(
                jnp.sum(
                    bspline(m + pme_order/2) * jnp.cos(2*jnp.pi*m*kpts_int[jnp.newaxis] / N),
                    axis = 0
                    ),
                axis = 1
                )
        V = jnp.linalg.det(box)
        S_k = jnp.fft.fftn(Q_mesh).flatten()
        # for electrostatic, need to exclude gamma point
        # for dispersion, need to include gamma point
        if not gamma:
            C_k = Ck_fn(kpts[3, 1:], kappa, V)
            E_k = C_k *  jnp.abs(S_k[1:] / theta_k[1:])**2
        else:
            C_k = Ck_fn(kpts[3, :], kappa, V)
            # debug
            # for i in range(1000):
            #     print('%15.8f%15.8f'%(jnp.real(C_k[i]), jnp.imag(C_k[i])))
            E_k = C_k * jnp.abs(S_k / theta_k)**2

        if not gamma: # doing electrics
            return jnp.sum(E_k) * DIELECTRIC
        else:
            return jnp.sum(E_k)

    if DO_JIT:
        return jit(pme_recip, static_argnums=())
    else:
        return pme_recip


def Ck_1(ksq, kappa, V):
    return 2*jnp.pi/V/ksq * jnp.exp(-ksq/4/kappa**2)

def Ck_6(ksq, kappa, V):
    x2 = ksq / 4 / kappa**2
    x = jnp.sqrt(x2)
    x3 = x2 * x
    exp_x2 = jnp.exp(-x2)
    f = (1 - 2*x2)*exp_x2 + 2*x3*sqrt_pi*jsp.special.erfc(x)
    return sqrt_pi*jnp.pi/2/V*kappa**3 * f / 3

def Ck_8(ksq, kappa, V):
    x2 = ksq / 4 / kappa**2
    x = jnp.sqrt(x2)
    x4 = x2 * x2
    x5 = x4 * x
    exp_x2 = jnp.exp(-x2)
    f = (3 - 2*x2 + 4*x4)*exp_x2 - 4*x5*sqrt_pi*jsp.special.erfc(x)
    return sqrt_pi*jnp.pi/2/V*kappa**5 * f / 45

def Ck_10(ksq, kappa, V):
    x2 = ksq / 4 / kappa**2
    x = jnp.sqrt(x2)
    x4 = x2 * x2
    x6 = x4 * x2
    x7 = x6 * x
    exp_x2 = jnp.exp(-x2)
    f = (15 - 6*x2 + 4*x4 - 8*x6)*exp_x2 + 8*x7*sqrt_pi*jsp.special.erfc(x)
    return sqrt_pi*jnp.pi/2/V*kappa**7 * f / 1260


def validation(pdb):
    jnp.set_printoptions(precision=32, suppress=True)
    xml = 'mpidwater.xml'
    pdbinfo = read_pdb(pdb)
    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    n_atoms = len(serials)

    atomTemplate, residueTemplate = read_xml(xml)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )
    Q = jnp.array(Q)
    Q_local = convert_cart2harm(Q, 2)
    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)


    # invoke pme energy calculator
    # energy_pme(positions, box, Q_local, axis_type, axis_indices, nbr.idx, 2)
    lmax = 2
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    # for debugging
    kappa = 0.657065221219616
    construct_local_frames_fn = generate_construct_local_frames(axis_type, axis_indices)
    local_frames = construct_local_frames_fn(positions, box)
    Q_global = rot_local2global(Q_local, local_frames, lmax)

    pme_order = 6
    energy_force_pme_recip = value_and_grad(generate_pme_recip(Ck_1, kappa, False, pme_order, K1, K2, K3, lmax))
    energy_force_pme_recip(positions, box, Q_global)
    print('ok')
    E, F = energy_force_pme_recip(positions, box, Q_global)
    print(E)

    # construct the C list
    c_list = np.zeros((3,n_atoms))
    nmol=int(n_atoms/3)
    for i in range(nmol):
        a = i*3
        b = i*3+1
        c = i*3+2
        c_list[0][a]=37.19677405
        c_list[0][b]=7.6111103
        c_list[0][c]=7.6111103
        c_list[1][a]=85.26810658
        c_list[1][b]=11.90220148
        c_list[1][c]=11.90220148
        c_list[2][a]=134.44874488
        c_list[2][b]=15.05074749
        c_list[2][c]=15.05074749
    energy_force_d6_recip = value_and_grad(generate_pme_recip(Ck_6, kappa, True, pme_order, K1, K2, K3, 0))
    energy_force_d8_recip = value_and_grad(generate_pme_recip(Ck_8, kappa, True, pme_order, K1, K2, K3, 0))
    energy_force_d10_recip = value_and_grad(generate_pme_recip(Ck_10, kappa, True, pme_order, K1, K2, K3, 0))
    E6, F6 = energy_force_d6_recip(positions, box, c_list[0, :, jnp.newaxis])
    E8, F6 = energy_force_d8_recip(positions, box, c_list[1, :, jnp.newaxis])
    E10, F10 = energy_force_d10_recip(positions, box, c_list[2, :, jnp.newaxis])
    print('ok')
    E6, F6 = energy_force_d6_recip(positions, box, c_list[0, :, jnp.newaxis])
    E8, F6 = energy_force_d8_recip(positions, box, c_list[1, :, jnp.newaxis])
    E10, F10 = energy_force_d10_recip(positions, box, c_list[2, :, jnp.newaxis])
    print(E6, E8, E10)
    print(E6 + E8 + E10)
    return


# validation code
if __name__ == '__main__':
    validation(sys.argv[1])
