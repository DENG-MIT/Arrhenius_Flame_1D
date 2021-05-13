if pwd()[end - 2:end] != "src"
    cd("src")
end

using Arrhenius
using LinearAlgebra
using ForwardDiff
using ForwardDiff:jacobian
using SparseArrays
using PyCall
using Plots
using Test

mech = "../mechanism/1S_CH4_MP1.yaml"
reactants = "CH4:0.5, O2:1.0, N2:3.76"

# mech = "../mechanism/h2o2.yaml"
# reactants = "H2:2.0, O2:1.0, AR:3.76"

## Call Cantera
ct = pyimport("cantera");
ct_gas = ct.Solution(mech);

# Simulation parameters
p = ct.one_atm  # pressure [Pa]
Tin = 300.0  # unburned gas temperature [K]
ct_gas.TPX = Tin, p, reactants

# Flame object
width = 0.1;  # m
f = ct.FreeFlame(ct_gas, width=width);
f.set_refine_criteria(ratio=3, slope=0.1, curve=0.1, prune=0.005);

f.solve(loglevel=1, auto=true);
f.velocity[1]

ct_sens = f.get_flame_speed_reaction_sensitivities();

gas = CreateSolution(mech);

ns = gas.n_species
nr = gas.n_reactions

include("flame_1d.jl")

cal_wdot = Wdot(f.P);
p = zeros(nr);
z = f.grid;
yall = vcat(f.Y, f.T');
mdot0 = f.density[1] * f.velocity[1];
yv = vcat(reshape(yall, :, 1), mdot0);
yL = vcat(@view(f.Y[:, 1]), f.T[1]);
ind_f = findfirst(f.T .> 1200.0);
T_f = f.T[ind_f];

Fv = residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f=T_f);

@time Fy = jacobian(yv -> residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f=T_f), yv);
@time Fp = jacobian(p -> residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f=T_f), p);

dydp = Fy \ Fp;

sens = - @view(dydp[end, :]) ./ mdot0;

hcat(sens, ct_sens)

@test sens ≈ ct_sens atol = 1.e-2

cos = dot(sens ./ norm(sens), ct_sens ./ norm(ct_sens))