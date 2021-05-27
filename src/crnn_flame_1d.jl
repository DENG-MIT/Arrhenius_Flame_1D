
struct Wdot <: Function
    P::Float64
end

function (cal_wdot::Wdot)(u, p)
    Y = @view(u[1:ns])
    T = u[end]
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    P = cal_wdot.P
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)

    vk, w_in_f, w_in_b, w_in_E, w_in_A = p2vec(p)

    _kf = @. exp(w_in_A - w_in_E * 4184.0 * 1000.0 / R / T)

    ΔS_R = vk * S0 / R
    ΔH_RT = vk * h_mole / (R * T)
    vkG = log(one_atm / R / T) .* sum(eachcol(vk))
    Keq =
        @. exp(ΔS_R - ΔH_RT + vkG)
    _kr = @. _kf / Keq

    u_in = @. log(clamp(C, 1.e-20, Inf))

    w_in_x_f = w_in_f * u_in;
    w_in_x_b = w_in_b * u_in;

    wdot = vk' * @. (exp(w_in_x_f) * _kf - exp(w_in_x_b) * _kr);

    return wdot
end

function modify_gas(ct_gas, p)
    
    list_r = []

    vk, w_in_f, w_in_b, w_in_E, w_in_A = p2vec(p)

    for i in 1:nr_crnn
        reactants = Dict()
        products = Dict()
        for j in 1:ns
            vks = vk[i, j]
            if vks < 0.0
                reactants[gas.species_names[j]] = -vks
            else
                products[gas.species_names[j]] = vks
            end
        end

        A = exp(w_in_A[i])
        b = 0.0
        E = w_in_E[i] * 4184.0 * 1000.0

        r1 = ct.ElementaryReaction(reactants, products)
        r1.rate = ct.Arrhenius(A, b, E)
        # r1.reversible = false

        push!(list_r, r1)
    end

    ct_gasm = ct.Solution(thermo="IdealGas", kinetics="GasKinetics", transport="Mix",
                          species=ct_gas.species(), reactions=list_r)

    return ct_gasm
end

EF = 30.0
AF = 15.0
function p2vec(p)
    _p = reshape(p, :, nse + 2)
    vk = _p[:,1:nse] * E_null
    # vk[1, :] .= [-1.5, 2.0, -1.0, 1.0, 0, 0]
    # vk[2, :] .= [-0.5, 0.0, 0.0, -1.0, 1.0, 0]
    w_in_f = clamp.(-vk, 0, 2.5);
    w_in_b = clamp.(vk, 0, 2.5);
    w_in_E = _p[:, end - 1] .+ EF
    w_in_A = _p[:, end] .+ AF
    return vk, w_in_f, w_in_b, w_in_E, w_in_A
end

function display_p(p)
    vk, w_in_f, w_in_b, w_in_E, w_in_A = p2vec(p)
    println("\n species (col) reaction (row)")
    println(gas.species_names)
    show(stdout, "text/plain", round.(hcat(vk, w_in_E, w_in_A), digits=3))
end
# display_p(p)

function init_p()
    p = randn(nr_crnn * (nse + 2))
    _p = reshape(p, :, nse + 2)
    # H2 / H / O/ O2 / OH / H2O / HO2 / H2O2 / AR
    vref = [-1.0, 0, 0, -0.5, 0, 1.0, 0, 0, 0]
    for i in 1:nr_crnn
        _p[i, 1:nse] .= (vref \ E_null')'  .* norm(vref)^2
        
        # if i > 2
        #     _p[i, 1:nse] .+= randn(nse) .* 0.01
        # end
    end
    # _p[:, 1:end - 2] .*= 0.1
    # _p[:, end - 1] .+= 20.0
    # _p[:, end] .+= 20.0
    return vec(_p)
end

# p = init_p()
# vk, w_in_f, w_in_b, w_in_E, w_in_A = p2vec(p);
# ct_gasm = modify_gas(ct_gas, p)

p = init_p();
display_p(p)

phi = 1.0
mgas = modify_gas(ct_gas, p)
f = solve_flame(mgas, phi)

# plot(f.grid, f.T)


function cal_grad(phi, p)

    mgas = modify_gas(ct_gas, p)
    f = solve_flame(mgas, phi)

    cal_wdot = Wdot(f.P);
    z = f.grid;
    yall = vcat(f.Y, f.T');
    mdot0 = f.density[1] * f.velocity[1];
    yv = vcat(reshape(yall, :, 1), mdot0);
    yL = vcat(@view(f.Y[:, 1]), f.T[1]);

    Tf = 1100.0
    if f.T[end] < Tf + 1.0
        println("Warning: flame is not correct for phi = $phi maxT = $(f.T[end])")
    end

    ind_f = findfirst(f.T .> Tf);
    T_f = f.T[ind_f];

    Fy = jacobian(yv -> residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f=T_f), yv);
    Fp = jacobian(p -> residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f=T_f), p);

    dydp = Fy \ Fp;

    sens = - @view(dydp[end, :]) ./ mdot0 .* (f.velocity[1]);

    return f.velocity[1], sens
end


"""
gas: 
cal_wdot:
p: params
z: grid points
yv: 
yL: left boundary conditions
ind_f:
T_f:
Equation @ https://cantera.org/science/flames.html
Cantera source code: https://github.com/Cantera/cantera/blob/be6e971dc6436dbc2bf86e56a41cab4b511a3dc3/src/oneD/StFlow.cpp
"""
function residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f)

    ng = length(z)
    y = reshape(@view(yv[1:end - 1]), ns + 1, ng)
    mdot = yv[end]

    mY = @view(y[1:ns, :])
    mT = @view(y[ns + 1, :])

    m_η_mix = similar(mT)
    m_λ_mix = similar(mT)
    m_Dkm = similar(mY)
    m_X = similar(mY)
    m_u = similar(mT)
    m_ρ = similar(mT)
    m_mean_MW = similar(mT)
    m_cp_mass = similar(mT)
    m_cpk = similar(mY)
    m_h_mole = similar(mY)
    m_wdot = similar(mY) * p[1]

    mT_m = similar(mT)
    mY_m = similar(mY)

    m_Dkm_m = similar(mY)
    m_η_mix_m = similar(mT)
    m_λ_mix_m = similar(mT)
    z_m = similar(z)

    @. z_m[1:end - 1] = (z[2:end] + z[1:end - 1]) / 2.0
    @. mT_m[1:end - 1] = (mT[2:end] + mT[1:end - 1]) / 2.0
    @. @view(mY_m[:, 1:end - 1]) = (@view(mY[:, 2:end]) + @view(mY[:, 1:end - 1])) / 2.0

    for i = 1:ng
        T = mT[i]
        Y = @view(mY[:, i])
        mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
        ρ_mass = cal_wdot.P / R / T * mean_MW
        X = Y2X(gas, Y, mean_MW)
        η_mix, λ_mix, Dkm = mix_trans(gas, cal_wdot.P, T, X, mean_MW)

        m_η_mix[i] = η_mix
        m_λ_mix[i] = λ_mix
        m_Dkm[:, i] = Dkm
        m_X[:, i] = X
        m_ρ[i] = ρ_mass
        m_mean_MW[i] = mean_MW
        m_u[i] = mdot / ρ_mass

        _, cp_mass = get_cp(gas, T, X, mean_MW)
        m_cp_mass[i] = cp_mass
        m_cpk[:, i] = cal_cpmass(gas, T, 0.0, X)
        m_h_mole[:, i] = get_H(gas, T, Y, X)

        m_wdot[:, i] = cal_wdot(@view(y[1:ns + 1, i]), p)
    end

    for i = 1:ng - 1
        T = mT_m[i]
        Y = @view(mY_m[:, i])
        mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
        X = Y2X(gas, Y, mean_MW)
        η_mix, λ_mix, Dkm = mix_trans(gas, cal_wdot.P, T, X, mean_MW)

        m_Dkm_m[:, i] = Dkm
        m_η_mix_m[i] = η_mix
        m_λ_mix_m[i] = λ_mix
    end

    # update_dYdx
    dXdz = similar(m_X)
    @. @view(dXdz[:, 1:end - 1]) = (@view(m_X[:, 2:end]) - @view(m_X[:, 1:end - 1])) / 
                                   (@view(z[2:end]) - @view(z[1:end - 1]))'

    dYdz = similar(mY)
    @. @view(dYdz[:, 2:end]) = (@view(mY[:, 2:end]) - @view(mY[:, 1:end - 1])) / 
                               (@view(z[2:end]) - @view(z[1:end - 1]))'

    # update_diffusive_fluxes
    Wk_W = gas.MW * (1 ./ m_mean_MW')
    jk_star = @. -m_ρ' * Wk_W * m_Dkm_m * dXdz
    jk = jk_star .- mY .* sum(jk_star, dims=1)

    djkdz = similar(jk)
    @. @view(djkdz[:, 2:end - 1]) = (@view(jk[:, 2:end - 1]) - @view(jk[:, 1:end - 2])) / 
                                    (@view(z_m[2:end - 1]) - @view(z_m[1:end - 2]))'

    F_Y = mdot .* dYdz + djkdz - gas.MW .* m_wdot

    # update_dTdx
    
    Twdot = - sum(m_h_mole .* m_wdot, dims=1)'

    dTdz = similar(mT)
    @. dTdz[2:end] = (mT[2:end] - mT[1:end - 1]) / (z[2:end] - z[1:end - 1])

    dTdz_div = similar(mT)
    @. dTdz_div[1:end - 1] = (mT[2:end] - mT[1:end - 1]) / 
                            (z[2:end] - z[1:end - 1])
                            
    dTdz_div_λ = @. dTdz_div .* m_λ_mix_m

    divHeatFlux = similar(mT)
    @. divHeatFlux[2:end - 1] = 2.0 * (dTdz_div_λ[2:end - 1] - dTdz_div_λ[1:end - 2]) / 
                                (z[3:end] - z[1:end - 2])

    jk_m = similar(jk)
    @. @view(jk_m[:, 2:end - 1]) = (@view(jk[:, 2:end - 1]) + @view(jk[:, 1:end - 2])) / 2.0
    Tflux = - dTdz .* sum(jk_m .* m_cpk, dims=1)'

    Tconv = @. (mdot * m_cp_mass * dTdz)

    F_T = Twdot + divHeatFlux + Tflux - Tconv

    # Boundary conditions
    F_Y[:, 1] = yL[1:end - 1] .- mY[:, 1] .- jk[:, 1] / mdot
    F_Y[:, end] = (mY[:, end] - mY[:, end - 1]) / (z[end] - z[end - 1])'

    F_T[1] = mT[1] - yL[end]
    F_T[end] = (mT[end] - mT[end - 1]) / (z[end] - z[end - 1])

    F = vcat(F_Y, F_T')

    # Test code
    # @test m_cp_mass ≈ f.cp_mass rtol = 1.e-4
    # @test m_λ_mix ≈ f.thermal_conductivity rtol = 1.e-4
    # @test m_wdot ≈ f.net_production_rates rtol = 1.e-4
    # @test m_Dkm ≈ f.mix_diff_coeffs rtol = 1.e-3
    # HR = - sum(m_h_mole .* m_wdot, dims=1)'
    # @test HR ≈ f.heat_release_rate rtol = 1.e-4

    return vcat(F..., mT[ind_f] - T_f)
end