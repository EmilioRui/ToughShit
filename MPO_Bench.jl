using PyCall
using ITensors
push!(pyimport("sys")."path", pwd())


n_modes = 3
njuncs = 3
fock_trunc=3


np = pyimport("numpy")
mode_freqs = np.array([1 for x in 1:n_modes])
junc_freq = [1 for x in 1:njuncs]
f_zp =np.ones([n_modes,njuncs])

py_module = pyimport("Hamiltonian")




print("I m here")
H = np.matrix(py_module.black_box_hamiltonian(mode_freqs,junc_freq,f_zp,fock_trunc=fock_trunc))

sites = siteinds("Boson",n_modes,dim=fock_trunc)
H_operator = op(H, sites)

print("Ready to make it an MPO")
time_MPO_creation =@elapsed begin
    
H_MPO = MPO(H_operator,sites)
end

print("\n MPO creation time: ", time_MPO_creation)


time_MPO_splitbox =@elapsed begin
H_MPO_split = MPO(H_operator,sites,splitblocks=true)
end
print("\n MPO splitbox creation time: ", time_MPO_splitbox)


time_MPO_trunc =@elapsed begin
H_MPO_limited = MPO(H_operator,sites,cutoff=1e-10)
end
print("\n MPO trunc time: ", time_MPO_creation,'\n')
