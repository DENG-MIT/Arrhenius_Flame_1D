
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
    # _p = reshape(p, nr, 3)
    # kp = @. @views(exp(_p[:, 1] + _p[:, 2] * log(T) - _p[:, 3] * 4184.0 / R / T))
    kp = exp.(p)
    qdot = wdot_func(gas.reaction, T, C, S0, h_mole; get_qdot=true) .* kp
    return gas.reaction.vk * qdot
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