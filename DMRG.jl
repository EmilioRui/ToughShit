using PyCall
using ITensors
using JSON
push!(pyimport("sys")."path", pwd())


function MPO_generator(sites,n_modes,fock_trunc)
    np = pyimport("numpy")

   # f_zp =np.ones([n_modes,njuncs])

    py_module = pyimport("Matrix_construction")



    #print("I m here")
    H = np.matrix(py_module.matrix_construction(chip_name= "TAR-0012-01", num_modes= n_modes, cos_trunc=5, fock_trunc=fock_trunc))

    H_operator = op(H, sites)

    print("Ready to make it an MPO")

    time_MPO_splitbox =@elapsed begin
    H_MPO = MPO(H_operator,sites,splitblocks=true,cutoff = 1e-9)
    end
    print("\n MPO splitbox creation time: ", time_MPO_splitbox,'\n')
    return H_MPO
end


function dmrg_A_B(H_MPO,sites,n_eigs, nsweeps , maxdim , cutoff )
    eigenvectors::Vector{MPS} = []
    eigenvalues = []
    for i in 1:n_eigs   
        ψ0 = randomMPS(sites,linkdims=50) 
        if i == 1
            energy, ψ = dmrg(H_MPO ,ψ0; nsweeps, maxdim, cutoff)
        else
            energy, ψ = dmrg(H_MPO ,eigenvectors, ψ0; nsweeps, maxdim, cutoff,weight=1e10,noise=1e5 )
        end
        push!(eigenvalues,energy)
        push!(eigenvectors,ψ)
        display(inner(eigenvectors[end],eigenvectors[1]))
    end
    return eigenvalues,eigenvectors
end




function bench_dmrg(n_modes,fock_trunc)
    np = pyimport("numpy")

    sites = siteinds("Boson",n_modes,dim=fock_trunc)


    #Julia Bench
    time_Julia = @elapsed begin
    H_MPO = MPO_generator(sites,n_modes,fock_trunc)
    n_eigs=n_modes
    nsweeps=20
    maxdim = 200
    cutoff=1e-10
    evals_dmrg,eigenvectors = dmrg_A_B(H_MPO,sites,n_eigs, nsweeps, maxdim, cutoff)
    end

    #Qutip Bench
    evals_qutip = []
    time_qutip = @elapsed begin
        py_module = pyimport("Hamiltonian")
        evals_qutip = py_module.qutip_diag(mode_freqs,junc_freq,f_zp,fock_trunc,n_modes)
    end

    print("Julia needed: ",time_Julia," s\n Qutip needed ", time_qutip, " s")

    # print("\n Julia evals: \n", eigenvalues)
    # print("\n Qutip evals: \n", evals_qutip)
    add_result_to_file("DMRG_bench 2.json",time_qutip,time_Julia,n_modes,evals_qutip ,evals_dmrg)
end


function add_result_to_file(filename::String,time_qutip,time_Julia,n_modes,evals_qutip,evals_dmrg)

    new_data = Dict("n_modes"=>n_modes,"Qutip_time"=>time_qutip, "Julia_Time"=>time_Julia,"Qutip Evals"=>evals_qutip,"DMRG evals"=>evals_dmrg)
    data = Dict()
    try
    # read the contents of the file into a string
        json_str = read(filename, String)
        data = JSON.parse(json_str)

    catch 
        data = Dict("Simulations" => [])
    end
    
    push!(data["Simulations"],new_data)

    open(filename,"w") do f
    JSON.print(f, data)
    end
    pyimport("Hamiltonian").indent_json(filename)
end




n_modes_vec = [2,3,5]
fock_trunc = 8

for n_modes in n_modes_vec
    bench_dmrg(n_modes,fock_trunc)
end
