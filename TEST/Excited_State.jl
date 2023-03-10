using ITensors

let
  N = 20

  sites = siteinds("S=1/2",N)

  h = 4.0
  
  weight = 20*h # use a large weight
                # since gap is expected to be large


  #
  # Use the OpSum feature to create the
  # transverse field Ising model
  #
  # Factors of 4 and 2 are to rescale
  # spin operators into Pauli matrices
  #
  os = OpSum()
  for j=1:N-1
    os += -4,"Sz",j,"Sz",j+1
  end
  for j=1:N
    os += -2*h,"Sx",j;
  end
  H = MPO(os,sites)


  #
  # Make sure to do lots of sweeps
  # when finding excited states
  #
  nsweeps = 30
  maxdim = [10,10,10,20,20,40,80,100,200,200]
  cutoff = [1E-8]
  noise = [1E-6]

  #
  # Compute the ground state psi0
  #
  psi0_init = randomMPS(sites,linkdims=2)
  energy0,psi0 = dmrg(H,psi0_init; nsweeps, maxdim, cutoff, noise)

  println()

  #
  # Compute the first excited state psi1
  #
  psi1_init = randomMPS(sites,linkdims=2)
  energy1,psi1 = dmrg(H,[psi0],psi1_init; nsweeps, maxdim, cutoff, noise, weight)

  # Check psi1 is orthogonal to psi0
  @show inner(psi1,psi0)


  #
  # The expected gap of the transverse field Ising
  # model is given by Eg = 2*|h-1|
  #
  # (The DMRG gap will have finite-size corrections)
  #
  println("DMRG energy gap = ",energy1-energy0);
  println("Theoretical gap = ",2*abs(h-1));

  println()

  #
  # Compute the second excited state psi2
  #
  psi2_init = randomMPS(sites,linkdims=2)
  energy2,psi2 = dmrg(H,[psi0,psi1],psi2_init; nsweeps, maxdim, cutoff, noise, weight)

  # Check psi2 is orthogonal to psi0 and psi1
  @show inner(psi2,psi0)
  @show inner(psi2,psi1)

  return
end
