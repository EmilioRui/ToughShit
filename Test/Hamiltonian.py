import qutip
from qutip import basis, tensor
import numpy as np
_initial_missing = object()
def dot(ais, bis):
    """
    Dot product
    """
    return sum(ai*bi for ai, bi in zip(ais, bis))
def fact(n):
    ''' Factorial '''
    if n <= 1:
        return 1
    return n * fact(n-1)
def cos_approx(x, cos_trunc=5):
    """
    Create a Taylor series matrix approximation of the cosine, up to some order.
    """
    return sum((-1)**i * x**(2*i) / float(fact(2*i)) for i in range(2, cos_trunc + 1))
def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, sequence[, initial]) -> value
    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the sequence in the calculation, and serves as a default when the
    sequence is empty.
    """
    it = iter(sequence)
    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError("reduce() of empty sequence with no initial value") from None
    else:
        value = initial
    for element in it:
        value = function(value, element)
    return value
def epr_numerical_diagonalization(freqs, Ejs, ϕzpf,
             cos_trunc=8,
             fock_trunc=9,
             use_1st_order=False,
             return_H=False):
    '''
    Numerical diagonalizaiton for pyEPR. Ask Zlatko for details.
    :param fs: (GHz, not radians) Linearized model, H_lin, normal mode frequencies in Hz, length M
    :param ljs: (Henries) junction linerized inductances in Henries, length J
    :param fzpfs: (reduced) Reduced Zero-point fluctutation of the junction fluxes for each mode
                across each junction, shape MxJ
    :return: Hamiltonian mode freq and dispersive shifts. Shifts are in MHz.
             Shifts have flipped sign so that down shift is positive.
    '''
    freqs, Ejs, ϕzpf = map(np.array, (freqs, Ejs, ϕzpf))
    assert(all(freqs < 1E6)
           ), "Please input the frequencies in GHz. \N{nauseated face}"
    assert(all(Ejs < 1E6)
           ), "Please input the energies in GHz. \N{nauseated face}"
    Hs = black_box_hamiltonian(freqs * 1E9, Ejs.astype(np.float) * 1E9, ϕzpf,
                 cos_trunc, fock_trunc, individual=use_1st_order)
    f_ND, χ_ND, _, _ = make_dispersive(
        Hs, fock_trunc, ϕzpf, freqs, use_1st_order=use_1st_order)
    χ_ND = -1*χ_ND * 1E-6  # convert to MHz, and flip sign so that down shift is positive
    return (f_ND, χ_ND, Hs) if return_H else (f_ND, χ_ND)


def black_box_hamiltonian(fs, ejs, fzpfs, cos_trunc=5, fock_trunc=8, individual=False):
    r"""
    :param fs: Linearized model, H_lin, normal mode frequencies in Hz, length N
    :param ljs: junction linerized inductances in Henries, length M
    :param fzpfs: Zero-point fluctutation of the junction fluxes for each mode across each junction,
                 shape MxJ
    :return: Hamiltonian in units of Hz (i.e H / h)
    All in SI units. The ZPF fed in are the generalized, not reduced, flux.
    Description:
     Takes the linear mode frequencies, $\omega_m$, and the zero-point fluctuations, ZPFs, and
     builds the Hamiltonian matrix of $H_full$, assuming cos potential.
    """
    n_modes = len(fs)
    njuncs = len(ejs)
    fs, ejs, fzpfs = map(np.array, (fs, ejs, fzpfs))
    # ejs = fluxQ**2 / ljs
    fjs = ejs
    fzpfs = np.transpose(fzpfs)  # Take from MxJ  to JxM
    assert np.isnan(fzpfs).any(
    ) == False, "Phi ZPF has NAN, this is NOT allowed! Fix me. \n%s" % fzpfs
    assert np.isnan(ejs).any(
    ) == False, "Ejs has NAN, this is NOT allowed! Fix me."
    assert np.isnan(
        fs).any() == False, "freqs has NAN, this is NOT allowed! Fix me."
    assert fzpfs.shape == (njuncs, n_modes), "incorrect shape for zpf array, {} not {}".format(
        fzpfs.shape, (njuncs, n_modes))
    assert fs.shape == (n_modes,), "incorrect number of mode frequencies"
    assert ejs.shape == (njuncs,), "incorrect number of qubit frequencies"
    def tensor_out(op, loc):
        "Make operator <op> tensored with identities at locations other than <loc>"
        op_list = [qutip.qeye(fock_trunc) for i in range(n_modes)]
        op_list[loc] = op
        return reduce(qutip.tensor, op_list)
    a = qutip.destroy(fock_trunc)
    ad = a.dag()
    n = qutip.num(fock_trunc)
    mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
    mode_ns = [tensor_out(n, i) for i in range(n_modes)]
    def cos(x):
        return cos_approx(x, cos_trunc=cos_trunc)
    linear_part = dot(fs, mode_ns)
    cos_interiors = [dot(fzpf_row, mode_fields) for fzpf_row in fzpfs]
    nonlinear_part = dot(-fjs, map(cos, cos_interiors))
    if individual:
        return linear_part, nonlinear_part
    else:
        return linear_part + nonlinear_part
    
    
def make_dispersive(H, fock_trunc, fzpfs=None, f0s=None, chi_prime=False,
                    use_1st_order=False):
    r"""
    Input: Hamiltonian Matrix.
        Optional: phi_zpfs and normal mode frequncies, f0s.
        use_1st_order : deprecated
    Output:
        Return dressed mode frequencies, chis, chi prime, phi_zpf flux (not reduced), and linear frequencies
    Description:
        Takes the Hamiltonian matrix `H` from bbq_hmt. It them finds the eigenvalues/eigenvectors and  assigns quantum numbers to them --- i.e., mode excitations,  such as, for instance, for three mode, |0,0,0> or |0,0,1>, which correspond to no excitations in any of the modes or one excitation in the 3rd mode, resp.    The assignment is performed based on the maximum overlap between the eigenvectors of H_full and H_lin.   If this crude explanation is confusing, let me know, I will write a more detailed one :slightly_smiling_face:
        Based on the assignment of the excitations, the function returns the dressed mode frequencies $\omega_m^\prime$, and the cross-Kerr matrix (including anharmonicities) extracted from the numerical diagonalization, as well as from 1st order perturbation theory.
        Note, the diagonal of the CHI matrix is directly the anharmonicity term.
    """
    if hasattr(H, '__len__'):  # is it an array / list?
        [H_lin, H_nl] = H
        H = H_lin + H_nl
    else:  # make sure its a quanutm object
        assert type(
            H) == qutip.qobj.Qobj, "Please pass in either  a list of Qobjs or Qobj for the Hamiltonian"
    # print("Starting the diagonalization")
    evals, evecs = H.eigenstates()
    # print("Finished the diagonalization")
    evals -= evals[0]
    N = int(np.log(H.shape[0]) / np.log(fock_trunc))    # number of modes
    assert H.shape[0] == fock_trunc ** N
    def fock_state_on(d):
        ''' d={mode number: # of photons} '''
        return qutip.tensor(*[qutip.basis(fock_trunc, d.get(i, 0)) for i in range(N)])  # give me the value d[i]  or 0 if d[i] does not exist
    if use_1st_order:
        num_modes = N
        print("Using 1st O")
        def multi_index_2_vector(d, num_modes, fock_trunc):
            '''this function creates a vector representation a given fock state given the data for excitations per
                        mode of the form d={mode number: # of photons}'''
            return tensor([basis(fock_trunc, d.get(i, 0)) for i in range(num_modes)])
        def find_multi_indices(fock_trunc):
            '''this function generates all possible multi-indices for three modes for a given fock_trunc'''
            multi_indices = [{ind: item for ind, item in enumerate([i, j, k])} for i in range(fock_trunc)
                             for j in range(fock_trunc)
                             for k in range(fock_trunc)]
            return multi_indices
        def get_expect_number(left, middle, right):
            '''this function calculates the expectation value of an operator called "middle" '''
            return (left.dag()*middle*right).data.toarray()[0, 0]   
        def get_basis0(fock_trunc, num_modes):
            '''this function creates a basis of fock states and their corresponding eigenvalues'''
            multi_indices = find_multi_indices(fock_trunc)
            basis0 = [multi_index_2_vector(
                multi_indices[i], num_modes, fock_trunc) for i in range(len(multi_indices))]
            evalues0 = [get_expect_number(v0, H_lin, v0).real for v0 in basis0]
            return multi_indices, basis0, evalues0
        def closest_state_to(vector0):
            def PT_on_vector(original_vector, original_basis, pertub, energy0, evalue):
                '''this function calculates the normalized vector with the first order correction term
                   from the non-linear hamiltonian '''
                new_vector = 0 * original_vector
                for i in range(len(original_basis)):
                    if (energy0[i]-evalue) > 1e-3:
                        new_vector += ((original_basis[i].dag()*H_nl*original_vector).data.toarray()[
                                       0, 0])*original_basis[i]/(evalue-energy0[i])
                    else:
                        pass
                return (new_vector + original_vector)/(new_vector + original_vector).norm()
            [multi_indices, basis0, evalues0] = get_basis0(
                fock_trunc, num_modes)
            evalue0 = get_expect_number(vector0, H_lin, vector0)
            vector1 = PT_on_vector(vector0, basis0, H_nl, evalues0, evalue0)
            index = np.argmax([(vector1.dag() * evec).norm()
                               for evec in evecs])
            return evals[index], evecs[index]
    else:
        def closest_state_to(s):
            def distance(s2):
                return (s.dag() * s2[1]).norm()
            return max(zip(evals, evecs), key=distance)
    f1s = [closest_state_to(fock_state_on({i: 1}))[0] for i in range(N)]
    chis = [[0]*N for _ in range(N)]
    chips = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            d = {k: 0 for k in range(N)}       # put 0 photons in each mode (k)
            d[i] += 1
            d[j] += 1
            # load ith mode and jth mode with 1 photon
            fs = fock_state_on(d)
            ev, evec = closest_state_to(fs)
            chi = (ev - (f1s[i] + f1s[j]))
            chis[i][j] = chi
            chis[j][i] = chi
            if chi_prime:
                d[j] += 1
                fs = fock_state_on(d)
                ev, evec = closest_state_to(fs)
                chip = (ev - (f1s[i] + 2*f1s[j]) - 2 * chis[i][j])
                chips[i][j] = chip
                chips[j][i] = chip
    if chi_prime:
        return np.array(f1s), np.array(chis), np.array(chips), np.array(fzpfs), np.array(f0s)
    else:
        return np.array(f1s), np.array(chis), np.array(fzpfs), np.array(f0s)
    
        
def qutip_diag(mode_freqs,junc_freq,f_zp,fock_trunc,n_modes):
    H=black_box_hamiltonian(mode_freqs,junc_freq,f_zp,fock_trunc=fock_trunc)
    evals, evecs = H.eigenstates()
    return evals