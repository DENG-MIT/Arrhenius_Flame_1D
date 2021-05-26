if pwd()[end - 2:end] != "src"
    cd("src")
end

using LinearAlgebra
using ForwardDiff
using SparseArrays
using PyCall

# mech = "../mechanism/h2o2.yaml"
# reactants = "H2:2.0, O2:1.0, AR:3.76"

mech = "../mechanism/1S_CH4_MP1.yaml"
reactants = "CH4:0.5, O2:1.0, N2:3.76"

## Call Cantera
ct = pyimport("cantera")
ct_gas = ct.Solution(mech)


# Simulation parameters
p = ct.one_atm  # pressure [Pa]
Tin = 300.0  # unburned gas temperature [K]

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

r = ct.ElementaryReaction(Dict("CH4" => 1.0, "O2" => 2.0, "CO2" => 0.0), 
                          Dict("CO2" => 1.0, "H2O" => 2.0))
r.reversible = false
r.rate = ct.Arrhenius(1.1e+10 * (1e-3)^0.5 * 0.5, 0.0, 20000 * 1000 * 4.184)
r.orders = Dict("CH4" => 1.0, "O2" => 0.5)

gas2 = ct.Solution(thermo="IdealGas", kinetics="GasKinetics", 
                   transport="mixture-averaged",
                   species=ct_gas.species(), reactions=[r])
                   
gas2.TPX = Tin, p, reactants

f2 = ct.FreeFlame(gas2, width=width)
f2.set_refine_criteria(ratio=3, slope=0.07, curve=0.14, prune=0.01)
f2.from_solution_array(fsol)

f2.solve(loglevel=0, auto=false)
f2.velocity[1]
f2.show_stats()