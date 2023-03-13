using PyCall
using ITensors
using JSON
push!(pyimport("sys")."path", pwd())

function H_generator(chip_name= "TAR-0012-01",fock_trunc= 8)
    # construct the file path
    filename = "Chip_Data/" * chip_name * ".json"
    # open the file and read its contents into the data variable
    # data = open(filename) do file
    #     JSON.parse(read(file, String))
    # end
    data = JSON.parsefile(filename)

    Ej = (data["Ejs"]/(data["Njs"].*data["Njs"])).*1e9
    𝜑_zpf = data["zpf"]
    freq = data["freq"].*1e9

    M = length(freq)
    J= length(Ej)

    sites = siteinds("Boson", M, dim=fock_trunc)

    print("Creating linear part of your Hamiltonian ...")
    H_lin = OpSum()
    for m=1:M
        H_lin += freq[m],"N",m
    end
    print("Linear part of your Hamiltonian has been created in $time s")

    print("Creating nonlinear part of your Hamiltonian ...")
    H_nonlin = OpSum()
    for j=1:J
        𝜑_j = OpSum()
        for m=1:M
            𝜑_j += 𝜑_zpf[m][j],"Adag",m
            𝜑_j += 𝜑_zpf[m][j],"A",m
        end
        print("Flux operator 𝜑 of Josephson dipole $j has been created")
        I = op("Id",sites)
        𝜑_j² = 𝜑_j*𝜑_j
        cos_𝜑_j = 0.5*(expHermitian(𝜑_j, 1_i) + expHermitian(𝜑_j, -1_i)) - I + 0.5*𝜑_j²
        print("cos(𝜑) of Josephson dipole $j has been created")
        H_nonlin += Ej[j]*cos_𝜑_j
    end
    print("Non linear part of your Hamiltonian has been created in $time s")
    H_tot = H_lin + H_nonlin

    print("Generating MPO")
    H_MPO = MPO(H_tot,sites,splitblocks=true,cutoff = 1e-9)

    return H_tot , H_MPO
end


function MPO_generator(sites,n_modes,fock_trunc)
    np = pyimport("numpy")

   # f_zp =np.ones([n_modes,njuncs])

    py_module = pyimport("Matrix_construction")



    #print("I m here")
    H_qutip = py_module.matrix_construction(chip_name= "TAR-0012-01", num_modes= n_modes, cos_trunc=5, fock_trunc=fock_trunc)
    H = np.matrix(H_qutip)
    H_operator = op(H, sites)

    print("Ready to make it an MPO")

    time_MPO_splitbox =@elapsed begin
    H_MPO = MPO(H_operator,sites,splitblocks=true,cutoff = 1e-9)
    end
    print("\n MPO splitbox creation time: ", time_MPO_splitbox,'\n')
    return H_MPO,H_qutip
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
    time_H_creation = @elapsed begin
    H_MPO,H_matrix = MPO_generator(sites,n_modes,fock_trunc)
    end
    println("Hamiltonian creation time : ",time_H_creation, " s")

    

    time_Julia = @elapsed begin
    n_eigs=n_modes
    nsweeps=20
    maxdim = 200
    cutoff=1e-10
    evals_dmrg,eigenvectors = dmrg_A_B(H_MPO,sites,n_eigs, nsweeps, maxdim, cutoff)
    evals_dmrg -= evals_dmrg[1]
    end

    println("DMRG optimization time: ", time_Julia, " s")

    #Qutip Bench
    evals_qutip = []
    time_qutip = @elapsed begin
        py_module = pyimport("Hamiltonian")
        evals_qutip = py_module.qutip_diag(H_matrix,n_modes)
    end

    println("Qutip optimization time: ", time_qutip, " s")

    # print("\n Julia evals: \n", eigenvalues)
    # print("\n Qutip evals: \n", evals_qutip)
    add_result_to_file("DMRG_time_bench.json",time_H_creation,time_qutip,time_Julia,n_modes,evals_qutip ,evals_dmrg)
end


function add_result_to_file(filename::String,time_H_creation,time_qutip,time_Julia,n_modes,evals_qutip,evals_dmrg)

    #relative error is (eig_Julia - eig Qutip / eig_Qutip
    realtive_error = abs((evals_qutip - evals_dmrg)./evals_qutip)


    new_data = Dict("n_modes"=>n_modes,"Time_Qutip"=>time_qutip,
     "Time_Julia"=>time_Julia,"Time_H"=>time_H_creation,"Qutip_Evals"=>evals_qutip,"DMRG_Evals"=>evals_dmrg,
      "Relative_Error"=>realtive_error)
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


n_modes_vec = [2,3,4]
fock_trunc = 8

for n_modes in n_modes_vec
    bench_dmrg(n_modes,fock_trunc)
end
