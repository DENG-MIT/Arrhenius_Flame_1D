if pwd()[end - 2:end] != "src"
    cd("src")
end

using LinearAlgebra
using ForwardDiff
using SparseArrays
using PyCall

mech = "../mechanism/h2o2.yaml"

## Call Cantera
ct = pyimport("cantera")
ct_gas = ct.Solution(mech)


# Simulation parameters
p = ct.one_atm  # pressure [Pa]
Tin = 300.0  # unburned gas temperature [K]
reactants = "H2:2.0, O2:1.0, AR:3.76"

width = 0.03  # m

ct_gas.TPX = Tin, p, reactants

# Flame object
f = ct.FreeFlame(ct_gas, width=width)
f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14, prune=0.01)

f.solve(loglevel=0, auto=true)
f.velocity[1]
f.show_stats()

sens = f.get_flame_speed_reaction_sensitivities()

fsol = f.to_solution_array()
f.from_solution_array(fsol)

f.X

f.Y

f.grid