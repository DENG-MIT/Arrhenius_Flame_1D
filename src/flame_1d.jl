
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
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)
    # _p = reshape(p, nr, 3)
    # kp = @. @views(exp(_p[:, 1] + _p[:, 2] * log(T) - _p[:, 3] * 4184.0 / R / T))
    kp = exp.(p)
    qdot = wdot_func(gas.reaction, T, C, S0, h_mole; get_qdot=true) .* kp
    wdot = gas.reaction.vk * qdot
    return wdot
end

function dydz!(dy, z, y)
    @. dy[1:end - 1] = (y[2:end] - y[1:end - 1]) / (z[2:end] - z[1:end - 1])
    # dy[1] = (y[2] - y[1]) / (z[2] - z[1])
end

function dydzm!(dy, z, y)
    @. dy[:, 1:end - 1] = (y[:, 2:end] - y[:, 1:end - 1]) / (z[2:end] - z[1:end - 1])'
    # @. dy[:, 1] = (y[:, 2] - y[:, 1]) / (z[2] - z[1])
end

function dydz_central!(dy, z, y)

    hj = z[3:end] - z[2:end - 1]
    hjL = z[2:end - 1] - z[1:end - 2]
    c1 = @. hjL / hj / (hj + hjL)
    c2 = @. (hj - hjL) / hj / hjL
    c3 = @. hj / hjL / (hj + hjL)
    @. dy[2:end - 1] = c1 * y[3:end] + c2 * y[2:end - 1] - c3 * y[1:end - 2]

    dy[1] = (y[2] - y[1]) / (z[2] - z[1])
    dy[end] = (y[end] - y[end - 1]) / (z[end] - z[end - 1])
end

function dydzm_central!(dy, z, y)

    hj = z[3:end] - z[2:end - 1]
    hjL = z[2:end - 1] - z[1:end - 2]
    c1 = @. hjL / hj / (hj + hjL)
    c2 = @. (hj - hjL) / hj / hjL
    c3 = @. hj / hjL / (hj + hjL)

    @. dy[:, 2:end - 1] = c1' * y[:, 3:end] + c2' * y[:, 2:end - 1] - c3' * y[:, 1:end - 2]
    
    @. dy[:, 1] = (y[:, 2] - y[:, 1]) / (z[2] - z[1])
    @. dy[:, end] = (y[:, end] - y[:, end - 1]) / (z[end] - z[end - 1])
end


"""
p: params
grid: grid points
yL: left boundary conditions
Equation @ https://cantera.org/science/flames.html
"""
function residual(gas, cal_wdot, p, z, yv, yL, ind_f; T_f)

    ng = length(z)
    y = reshape(@view(yv[1:end - 1]), ny, ng)
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

    # m_ρ_m = similar(mT)
    # m_X_m = similar(mY)
    # m_mean_MW_m = similar(mT)
    m_Dkm_m = similar(mY)
    m_η_mix_m = similar(mT)
    m_λ_mix_m = similar(mT)

    @. mT_m[1:end - 1] = (mT[2:end] + mT[1:end - 1]) / 2.0
    @. mY_m[:, 1:end - 1] = (mY[:, 2:end] + mY[:, 1:end - 1]) / 2.0

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

        cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
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
    # dydzm!(dXdz, z, m_X)
    @. dXdz[:, 1:end - 1] = (m_X[:, 2:end] - m_X[:, 1:end - 1]) / (z[2:end] - z[1:end - 1])'

    dYdz = similar(mY)
    # dydzm!(dYdz, z, mY)
    @. dYdz[:, 2:end] = (mY[:, 2:end] - mY[:, 1:end - 1]) / (z[2:end] - z[1:end - 1])'

    # update_diffusive_fluxes
    Wk_W = gas.MW * (1 ./ m_mean_MW')
    jk_star = @. -m_ρ' * Wk_W * m_Dkm_m * dXdz
    jk = jk_star .- mY .* sum(jk_star, dims=1)

    z_m = similar(z)
    @. z_m[1:end - 1] = (z[2:end] + z[1:end - 1]) / 2.0
    djkdz = similar(jk)
    @. djkdz[:, 2:end - 1] = (jk[:, 2:end - 1] - jk[:, 1:end - 2]) / (z_m[2:end - 1] - z_m[1:end - 2])'

    # F1 = mdot .* dYdz
    # F2 = djkdz
    # F3 = - gas.MW .* m_wdot
    # F = F1 + F2 + F3
    # hcat(F1[:, ind_f], F2[:, ind_f], F3[:, ind_f], F[:, ind_f])

    F_Y = mdot .* dYdz + djkdz - gas.MW .* m_wdot


    # update_dTdx
    dTdz = similar(mT)
    # dydz_central!(dTdz, z, mT)
    @. dTdz[2:end] = (mT[2:end] - mT[1:end - 1]) / (z[2:end] - z[1:end - 1])

    divHeatFlux = similar(mT)
    # dydz_central!(dTdz0, z, dTdz .* m_λ_mix_m)
    dTdz_div = similar(mT)
    @. dTdz_div[1:end - 1] = (mT[2:end] - mT[1:end - 1]) / (z[2:end] - z[1:end - 1])
    dTdz_div_λ = @. dTdz_div .* m_λ_mix_m

    @. divHeatFlux[2:end - 1] = 2.0 * (dTdz_div_λ[2:end - 1] - dTdz_div_λ[1:end - 2]) / 
                                (z[3:end] - z[1:end - 2])

    jk_m = similar(jk)
    @. jk_m[:, 2:end - 1] = (jk[:, 2:end - 1] + jk[:, 1:end - 2]) / 2.0

    # dTdz1 = dTdz0 - dTdz .* sum(jk_m .* m_cpk, dims=1)'
    # dTdz2 = dTdz1 - sum(m_h_mole .* m_wdot, dims=1)'

    Twdot = - sum(m_h_mole .* m_wdot, dims=1)'
    Tflux = - dTdz .* sum(jk_m .* m_cpk, dims=1)'

    Tconv = @. (mdot * m_cp_mass * dTdz)
    F_T = Twdot + divHeatFlux + Tflux - Tconv
    hcat(Twdot, Tconv, Tflux, divHeatFlux, F_T)[ind_f, :]'

    # boundary conditions
    F_T[1] = mT[1] - yL[end]
    F_T[end] = (mT[end] - mT[end - 1]) / (z[end] - z[end - 1])
    # F_T[ind_f] = mT[ind_f] - T_f

    F_Y[:, 1] = yL[1:end - 1] .- mY[:, 1] .- jk[:, 1] / mdot
    F_Y[:, end] = (mY[:, end] - mY[:, end - 1]) / (z[end] - z[end - 1])'

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