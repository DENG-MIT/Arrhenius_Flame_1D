if pwd()[end - 2:end] != "src"
    cd("src")
end

using Arrhenius
using LinearAlgebra
using ForwardDiff
using SparseArrays
using PyCall
using Plots

mech = "../mechanism/h2o2.yaml"

## Call Cantera
ct = pyimport("cantera")
ct_gas = ct.Solution(mech)

# Simulation parameters
p = ct.one_atm  # pressure [Pa]
Tin = 300.0  # unburned gas temperature [K]
reactants = "H2:2.0, O2:1.0, AR:3.76"
ct_gas.TPX = Tin, p, reactants

# Flame object
width = 0.1  # m
f = ct.FreeFlame(ct_gas, width=width)
f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14, prune=0.01)

f.solve(loglevel=1, auto=true)
f.velocity[1]

ct_sens = f.get_flame_speed_reaction_sensitivities()

gas = CreateSolution(mech)

ns = gas.n_species
nr = gas.n_reactions

include("flame_1d.jl")

