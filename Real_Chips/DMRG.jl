using PyCall
using ITensors
push!(pyimport("sys")."path", pwd())


n_modes = 4
njuncs = 3
fock_trunc=6

function ground_state_dmrg(n_modes,fock_trunc)
    np = pyimport("numpy")

    py_module = pyimport("Matrix_construction.py")


    sites = siteinds("Boson",n_modes,dim=fock_trunc)




    print("I m here")
    H = np.matrix(py_module.matrix_construction(chip_name= "TAR-0012-01", cos_trunc=5, fock_trunc=fock_trunc) )

    H_operator = op(H, sites)

    print("Ready to make it an MPO")

    time_MPO_splitbox =@elapsed begin
    H_MPO = MPO(H_operator,sites,splitblocks=true)
    end
    print("\n MPO splitbox creation time: ", time_MPO_splitbox,'\n')


    ψ0 = randomMPS(sites,30)



    nsweeps = 100

    #maxdim = [10,20,100,100,200]
    #Limit for cutoff
    maxdim = 100
    cutoff = [1E-3]

    energy, ψ = dmrg(H_MPO,ψ0; nsweeps, maxdim, cutoff)
    return energy,ψ
end

function multivariate_dmrg(n_eigs)
    for i in 1:n_eigs
        ground_state_dmrg
    end
end