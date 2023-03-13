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
    print("\nLinear part of your Hamiltonian has been created in $time s")

    print("\n\nCreating nonlinear part of your Hamiltonian ...")
    H_nonlin = OpSum()
    for j=1:J
        𝜑_j = OpSum()
        for m=1:M
            𝜑_j += 𝜑_zpf[m][j],"Adag",m
            𝜑_j += 𝜑_zpf[m][j],"A",m
        end
        print("\n Flux operator 𝜑 of Josephson dipole $j has been created")
        # I = op("Id",sites)
        # 𝜑_j² = 𝜑_j*𝜑_j
        # cos_𝜑_j = 0.5*(expHermitian(𝜑_j, 1_i) + expHermitian(𝜑_j, -1_i)) - I + 0.5*𝜑_j².
        cos_𝜑_j = exp(𝜑_j)
        print("\n cos(𝜑) of Josephson dipole $j has been created")
        H_nonlin += cos_𝜑_j
    end
    print("\n Non linear part of your Hamiltonian has been created in $time s")
    H_tot = H_lin + H_nonlin

    print("\n Generating MPO")
    H_MPO = MPO(H_tot,sites,splitblocks=true,cutoff = 1e-9)

    return H_tot , H_MPO
end

H_generator()