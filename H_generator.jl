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
    š_zpf = data["zpf"]
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
        š_j = OpSum()
        for m=1:M
            š_j += š_zpf[m][j],"Adag",m
            š_j += š_zpf[m][j],"A",m
        end
        print("\n Flux operator š of Josephson dipole $j has been created")
        # I = op("Id",sites)
        # š_jĀ² = š_j*š_j
        # cos_š_j = 0.5*(expHermitian(š_j, 1_i) + expHermitian(š_j, -1_i)) - I + 0.5*š_jĀ².
        cos_š_j = exp(š_j)
        print("\n cos(š) of Josephson dipole $j has been created")
        H_nonlin += cos_š_j
    end
    print("\n Non linear part of your Hamiltonian has been created in $time s")
    H_tot = H_lin + H_nonlin

    print("\n Generating MPO")
    H_MPO = MPO(H_tot,sites,splitblocks=true,cutoff = 1e-9)

    return H_tot , H_MPO
end

H_generator()