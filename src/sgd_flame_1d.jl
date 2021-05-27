if pwd()[end - 2:end] != "src"
    cd("src")
end

using Arrhenius
using LinearAlgebra
using ForwardDiff
using ForwardDiff:jacobian
using Random
using Flux
using Flux.Optimise:update!
using Flux.Losses:mae
using BSON: @save, @load
using SparseArrays
using PyCall
using Plots
using Test
using ProgressBars, Printf
using Statistics

# mech = "../mechanism/2S_CH4_CM2.yaml"
# reactants = "CH4:0.5, O2:1.0, N2:3.76"

mech = "../mechanism/h2o2.yaml"
reactants = "H2:2.0, O2:1.0, AR:3.76"

gas = CreateSolution(mech);

ns = gas.n_species
nr = gas.n_reactions
nr_crnn = 10

E_ = gas.ele_matrix;
E_null = nullspace(E_)';
nse = size(E_null)[1];

function creat_flame(ct_gas, phi)
    # reactants = "CH4:$(phi * 0.5), O2:1.0, N2:3.76"
    reactants = "H2:$(phi * 2.0), O2:1.0, AR:3.76"
    ct_gas.TPX = 300.0, ct.one_atm, reactants
    f = ct.FreeFlame(ct_gas, width=width);
    return f
end

function solve_flame(ct_gas, phi)
    f = creat_flame(ct_gas, phi)
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.1, prune=0.001);
    f.solve(loglevel=0, auto=true);
    return f
end

## Call Cantera
ct = pyimport("cantera");
ct_gas = ct.Solution(mech);
width = 0.1;  # m

n_exp = 10;
span_phi = [0.6, 1.5];
l_phi = span_phi[1]:(span_phi[2] - span_phi[1]) / (n_exp - 1):span_phi[2]

l_fsol = Dict()
l_SL = zeros(n_exp)
for i in 1:n_exp
    phi = l_phi[i]
    f = solve_flame(ct_gas, phi)
    l_fsol[phi] = f.to_solution_array()
    l_SL[i] = f.velocity[1]
    @printf("%4d %.2f %.2e\n", i, phi, l_SL[i])
end

include("crnn_flame_1d.jl")

# p = randn(nr * 3) .* 0.1;
p = init_p()

for i in 1:n_exp
    phi = l_phi[i]
    sl, sens = cal_grad(phi, p)
    @printf("%4d %.2f %.2e %.2e %.2e \n", i, phi, l_SL[i], sl, norm(sens))
end

l_epoch = ones(n_exp);
grad_norm = ones(n_exp);
l_loss = []

opt = ADAMW(0.01, (0.9, 0.999), 1.e-6);

epochs = ProgressBar(1:100)
for epoch in epochs
    global p
    l_epoch .= 1.0
    grad_norm .= 1.0
    for i in randperm(n_exp)
        phi = l_phi[i]
        sl, sens = cal_grad(phi, p)
        grad = 2 * (sl - l_SL[i]) * sens .* 1e4
        grad_norm[i] = norm(grad)
        update!(opt, p, grad)
        l_epoch[i] = (sl - l_SL[i])^2  .* 1e4
        @printf("%4d %.2f %.2e %.2e %.2e \n", i, phi, l_SL[i], sl, norm(grad))
    end
    push!(l_loss, mean(l_epoch))

    set_description(
        epochs,
        string(
            @sprintf("loss %.4e pnorm %.2e gnorm %.2e",
            l_loss[epoch], norm(p), mean(grad_norm)
            )
        ),
    )
    if epoch % 1 == 0
        plt = Plots.plot(l_loss)
        png(plt, "./loss.png")
        display_p(p)
    end
end