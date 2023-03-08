using PyCall
using ITensors
push!(pyimport("sys")."path", pwd())

#From qutip importing the big matrix named H
n_modes = 5
njuncs = 5
fock_trunc=9


np = pyimport("numpy")
mode_freqs = np.array([1 for x in 1:n_modes])
junc_freq = [1 for x in 1:njuncs]
f_zp =np.ones([n_modes,njuncs])

py_module = pyimport("Hamiltonian")



H = np.matrix(py_module.black_box_hamiltonian(mode_freqs,junc_freq,f_zp,fock_trunc=fock_trunc))

sites = siteinds("Boson",n_modes,dim=fock_trunc)
H_operator = op(H, sites)

H_MPO = MPO(H_operator,sites)
ψ0 = randomMPS(sites,20)



nsweeps = 10
maxdim = [10,20,100,100,200]
cutoff = [1E-10]

energy, ψ = dmrg(H_MPO,ψ0; nsweeps, maxdim, cutoff)
