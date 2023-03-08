using PyCall
using ITensors
push!(pyimport("sys")."path", pwd())


function MPO_generator(n_modes,njuncs,fock_trunc)
    np = pyimport("numpy")
    global mode_freqs,junc_freq,f_zp

   # f_zp =np.ones([n_modes,njuncs])

    py_module = pyimport("Hamiltonian")

    global sites


    print("I m here")
    H = np.matrix(py_module.black_box_hamiltonian(mode_freqs,junc_freq,f_zp,fock_trunc=fock_trunc))

    H_operator = op(H, sites)

    print("Ready to make it an MPO")

    time_MPO_splitbox =@elapsed begin
    H_MPO = MPO(H_operator,sites,splitblocks=true)
    end
    print("\n MPO splitbox creation time: ", time_MPO_splitbox,'\n')
    return H_MPO
end


function dmrg_A_B(H_MPO;n_eigs, nsweeps , maxdim , cutoff )
    eigenvectors = []
    eigenvalues = []
    for i in 1:n_eigs   
        ψ0 = randomMPS(sites,30) 
        energy, ψ = dmrg(H_MPO,ψ0; nsweeps, maxdim, cutoff)
        push!(eigenvalues,energy)
        push!(eigenvectors,ψ)
    end
    return eigenvalues,eigenvectors
end


n_modes = 5
njuncs = 3
fock_trunc = 3
np = pyimport("numpy")

mode_freqs = np.array([rand(10^8:10^10) for x in 1:n_modes])
junc_freq = [rand(10^8:10^10) for x in 1:njuncs]
f_zp = np.array(rand(10^8:10^10,(n_modes,njuncs)))
sites = siteinds("Boson",n_modes,dim=fock_trunc)



#Julia Bench
time_Julia = @elapsed begin
H_MPO = MPO_generator(n_modes,njuncs,fock_trunc)
eigenvalues,eigenvectors = dmrg_A_B(H_MPO,n_eigs=n_modes, nsweeps=20 , maxdim=50 , cutoff=1e-10)
end

#Qutip Bench
evals_qutip = []
time_qutip = @elapsed begin
    py_module = pyimport("Hamiltonian")
    evals_qutip = py_module.qutip_diag(mode_freqs,junc_freq,f_zp,fock_trunc,n_modes)
end

print("Julia needed: ",time_Julia," s\n Qutip needed ", time_qutip, " s")

print("\n Julia evals: \n", eigenvalues)
print("\n Qutip evals: \n", evals_qutip)