import qutip
from qutip import basis, tensor, Qobj

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

def exp_approx(x, exp_trunc=5):
    """
    Create a Taylor series matrix approximation of the cosine, up to some order.
    """
    return sum((-1)**i * x**i / float(fact(i)) for i in range(4, 2*cos_trunc + 1))

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


def black_box_hamiltonian(fs, ejs, fzpfs, cos_trunc=5, exp_trunc=5, fock_trunc=8, individual=False):
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
        "We can't tensor operators of size greater than 10⁹"
        op_list = [qutip.qeye(fock_trunc) for i in range(n_modes)]
        op_list[loc] = op
        return reduce(qutip.tensor, op_list)

    def tensor_exp(ops):
        #ops should a list of operators
        op_list = [exp_approx(ops[i], exp_trunc=exp_trunc) for i in range(n_modes)]
        return reduce(qutip.tensor, op_list)
    
    
    a = qutip.destroy(fock_trunc)
    ad = a.dag()
    n = qutip.num(fock_trunc)


    mode_ns = [tensor_out(n, i) for i in range(n_modes)]
    # print('This array took',f'{time.time() - start2}')

    def cos(x):
        "Big powers of big tensors may cause issues. Matrix of size 10⁵ takes too much time to give power of 10"
        return cos_approx(x, cos_trunc=cos_trunc)
    
    def cos_exp(ops):

        op1 = list(map(lambda x: x*complex(0,1), ops))      # Kronecker product is bilinear
        op2 = list(map(lambda x: -x*complex(0,1), ops))
        return 0.5*(tensor_exp(op1) + tensor_exp(op2))

    linear_part = dot(fs, mode_ns)
    print('Linear part of your H has been built!\U0001F973')

    print('\nThe construction of non linear part of H has started, patience is needed \U000023F3')

    print('\n   We start by constructing a array of cos(𝜑)')
    start = time.time()
    cos_𝜑_js = [cos_exp(list(map(lambda x: (a + ad)*x, fzpf_row))) for fzpf_row in fzpfs]       # Kronecker product is bilinear
    print('   cos(𝜑) array took',f'Time: {time.time() - start}')
    # cos_interiors = [dot(fzpf_row, mode_fields) for fzpf_row in fzpfs]
    # print('   cos_interiors done !')
    # print('   Adding cos function to it')
    nonlinear_part = dot(-fjs, cos_𝜑_js)

    H_tot = linear_part + nonlinear_part

    print('\n Benchmarking H_nonlin')

    mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
    cos_interiors = [dot(fzpf_row, mode_fields) for fzpf_row in fzpfs]
    print('  Adding cos')
    cos_𝜑_js_old= list(map(cos, cos_interiors))
    nonlinear_part_old = dot(-fjs, map(cos, cos_interiors))


    diff= np.array(nonlinear_part) - np.array(nonlinear_part_old)
    relative_difference = diff*100/np.array(nonlinear_part_old)
    relative_difference= np.abs(diff)
    max_relative_diff= relative_difference.max()

    print(f'\n Maximum relative difference is {max_relative_diff}%')

    print('\n old H_lin cos')
    print(cos_𝜑_js_old[1])
    # print(nonlinear_part_old)

    print('\n new H_lin')
    print(cos_𝜑_js[1])
    # print(nonlinear_part)

    # sr= np.count_nonzero(H_tot)*100/H_tot.shape[0]**2

    # print(f'Sparsity ratio of total Hamiltonian is {sp}%')
    print('\nYour Hamiltonian is ready to be diagonalized \U0001F37B')

    if individual:
        return linear_part, nonlinear_part
    else:
        return H_tot