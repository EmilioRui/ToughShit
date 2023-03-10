import numpy as np
from abdb import *
import qutip
from qutip import basis, tensor
import json


def matrix_construction(chip_name= 'TAR-0012-01', num_modes= 4, cos_trunc=5, fock_trunc=8):
    '''num_modes should be < 5. Otherwise the matrix contruction will fail'''

        ###########
    ##### Part 1 : Extract information from ABDB #######################################################################
        ###########

    

        ###########
    ##### Part 2 : Some preliminary functions for BBH function #########################################################
        ###########

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

        ###########
    ##### Part 3 : BBH function #############################################################################################
        ###########

    def black_box_hamiltonian(fs, ejs, fzpfs, cos_trunc=5, fock_trunc=8):
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

        return linear_part + nonlinear_part

    filename = 'Chip_Data' + chip_name + '.json'
    with open (filename, 'r') as f:
        data = json.load(f)
    
    Njs = data['Njs']
    Ejs = data['Ejs']
    zpf = data['zpf']
    freq = data['freq']

    
    Ejs = np.array(Ejs)/(np.array(Njs)**2)
    Ejs= np.array(Ejs)

    return black_box_hamiltonian(fs= np.array(freq[:num_modes]) * 1E9, 
                                ejs= Ejs.astype(np.float64) * 1E9, 
                                fzpfs= np.array(zpf[:num_modes]), 
                                cos_trunc=5, fock_trunc=8)


matrix_construction(chip_name= 'TAR-0012-01', num_modes= 4, cos_trunc=5, fock_trunc=8)