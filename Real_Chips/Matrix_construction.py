from abdb import *
import qutip
from qutip import basis, tensor



def matrix_construction(chip_name= 'TAR-0012-01', cos_trunc=5, fock_trunc=8):

    var = Variation.query().get(f'{chip_name}')
    sim = var.simulations.first()

    #remove side junctions and their energies
    idx = []
    new_je = []
    new_jj = []
    for i, je in enumerate(sim.data.josephson_elements):
        if '_jct' in je:
            idx.append(i)
        else:
            new_je.append(je)
            new_jj.append(sim.data.josephson_energies[i])

    # ZPF
    tmp_zpf = []

    for j, _zpf in enumerate(sim.data.phi_ZPF):
        tmp_zpf.append([])
        for i, _modezpf in enumerate(_zpf):
            if i not in idx:
                tmp_zpf[j].append(_modezpf)
        # tmp_zpf[j] = np.array(tmp_zpf[j])

    zpf = tmp_zpf

    # Energies
    Ejs = []
    Njs = []
    for i, ej in enumerate(new_jj):
        if isinstance(ej,TransmonEnergy):
            # if '_jct' not in sim.data.josephson_elements[i]:
            Ejs.append(ej.EJ/1e9)
            Njs.append(1)
        if isinstance(ej,ArrayEnergy):
            Ejs.append(ej.EL/1e9)
            Njs.append(ej.n_array)
        if isinstance(ej,ATSEnergy):
            Ejs.append(ej.EL/1e9)
            Njs.append(ej.n_array)

    # Freq
    freq_c = np.array(sim.data.frequencies)
    chispice = sim.data.chi

    freq = []
    for i, f in enumerate(freq_c):
        freq.append( (f + np.sum(chispice[i])) * 1e-9 )

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

    def tensor_out(op, loc):
        "Make operator <op> tensored with identities at locations other than <loc>"
        op_list = [qutip.qeye(fock_trunc) for i in range(n_modes)]
        op_list[loc] = op
        return reduce(qutip.tensor, op_list)

    def cos(x):
        return cos_approx(x, cos_trunc=cos_trunc)

    freqs= [*freq]

    Ejs=np.array(Ejs)/(np.array(Njs)**2)

    φzpf=[*zpf]

    freqs, Ejs, φzpf = map(np.array, (freqs, Ejs, φzpf))

    fs= freqs * 1E9
    ejs= Ejs.astype(np.float) * 1E9
    fzpfs= φzpf

    n_modes = len(fs)
    njuncs = len(ejs)
    fs, ejs, fzpfs = map(np.array, (fs, ejs, fzpfs))
    # ejs = fluxQ**2 / ljs
    fjs = ejs

    fzpfs = np.transpose(fzpfs)

    
    a = qutip.destroy(fock_trunc)
    ad = a.dag()
    n = qutip.num(fock_trunc)
    mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
    mode_ns = [tensor_out(n, i) for i in range(n_modes)]


    linear_part = dot(fs, mode_ns)
    cos_interiors = [dot(fzpf_row, mode_fields) for fzpf_row in fzpfs]
    nonlinear_part = dot(-fjs, map(cos, cos_interiors))

    return linear_part + nonlinear_part
    