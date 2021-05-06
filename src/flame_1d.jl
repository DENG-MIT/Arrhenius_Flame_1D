
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
    # Ydot = wdot / ρ_mass .* gas.MW
    # Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass
    Ydot = wdot
    Tdot = - sum(h_mole .* wdot, dims=1)
    du = vcat(Ydot, Tdot)
    return du
end

function get_cpk(gas, T, X, mean_MW)
    cp_T = [1.0, T, T^2, T^3, T^4]
    if T <= 1000.0
        cp = @view(gas.thermo.nasa_low[:, 1:5]) * cp_T
    else
        cp = @view(gas.thermo.nasa_high[:, 1:5]) * cp_T
    end
    # TODO: not sure if inplace operation will be an issue for AD
    if !gas.thermo.isTcommon
        ind_correction = @. (T > 1000.0) & (T < gas.thermo.Trange[:, 2])
        cp[ind_correction] .=
            @view(gas.thermo.nasa_low[ind_correction, 1:5]) * cp_T
    end
    return cp
end

function dydz!(dy, z, y)
    @. dy[2:end] = (y[2:end] - y[1:end - 1]) / (z[2:end] - z[1:end - 1])
    dy[1] = (y[2] - y[1]) / (z[2] - z[1])
end

function dydzm!(dy, z, y)
    @. dy[:, 2:end] = (y[:, 2:end] - y[:, 1:end - 1]) / (z[2:end] - z[1:end - 1])'
    @. dy[:, 1] = (y[:, 2] - y[:, 1]) / (z[2] - z[1])
end


"""
p: params
grid: grid points
y: solution variables: [Y, T, u] {ns+2, ng}
yL: left boundary conditions
Equation @ https://cantera.org/science/flames.html
"""
function residual(gas, cal_wdot, p, z, yv, yL, ind_f, T_f)

    ng = length(z)
    y = reshape(yv, ng, ny)'

    mY = @view(y[1:ns, :])
    mT = @view(y[ns + 1, :])
    mu = @view(y[ns + 2, :])

    m_η_mix = similar(mT)
    m_λ_mix = similar(mT)
    m_Dkm = similar(mY)
    m_X = similar(mY)
    m_ρ = similar(mT)
    m_mean_MW = similar(mT)
    m_cp_mole = similar(mT)
    m_cp_mass = similar(mT)
    m_cpk = similar(mY)
    m_h_mole = similar(mY)
    m_wdot = similar(y[1:ns + 1, :]) * p[1]

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

        cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
        m_cp_mole[i] = cp_mole
        m_cp_mass[i] = cp_mass
        m_cpk[:, i] = get_cpk(gas, T, X, mean_MW)
        m_h_mole[:, i] = get_H(gas, T, Y, X)

        m_wdot[:, i] = cal_wdot(@view(y[1:ns + 1, i]), p)
    end

    dXdz = similar(m_X)
    dydzm!(dXdz, z, m_X)
    dYdz = similar(mY)
    dydzm!(dYdz, z, mY)

    # update_diffusive_fluxes
    Wk_W = gas.MW * (1 ./ m_mean_MW')
    jk_star = @. - m_ρ' * Wk_W * m_Dkm * dXdz
    jk = jk_star - mY .* sum(jk_star, dims=1)

    djkdz = similar(jk)
    dydzm!(djkdz, z, jk)

    # update_dYdx
    # F1 = (m_ρ .* mu)' .* dYdz
    # F2 = djkdz
    # F3 = - gas.MW .* @view(m_wdot[1:end - 1, :])
    # F = F1 + F2 + F3

    F_Y = (m_ρ .* mu)' .* dYdz + djkdz - gas.MW .* @view(m_wdot[1:end - 1, :])

    dTdz = similar(mT)
    dydz!(dTdz, z, mT)

    dTdz0 = similar(mT)
    dydz!(dTdz0, z, dTdz .* m_λ_mix)

    dTdz1 = dTdz0 - dTdz .* sum(jk .* m_cpk, dims=1)'
    dTdz2 = dTdz1 + @view(m_wdot[end, :])

    # update_dTdx
    # F1 = dTdz2
    # F2 = @. (m_ρ * mu * m_cp_mass * dTdz)
    # F = @. dTdz2 - (m_ρ * mu * m_cp_mass * dTdz)

    F_T = @. dTdz2 - (m_ρ * mu * m_cp_mass * dTdz)

    # boundary conditions
    F_T[1] = mT[1] - yL[end]
    F_T[end] = dTdz[end]

    F_Y[:, 1] = yL[1:end - 1] .- mY[:, 1] .- jk[:, 1] / (m_ρ[1] * mu[1])
    F_Y[:, end] = dYdz[:, end]

    F_u = F_T .* 0.0
    F_u[1] = mT[ind_f] - T_f

    dudz = similar(mu)
    dydz!(dudz, z, m_ρ .* mu)

    F = vcat(F_Y, F_T', dudz')

    # Test code
    # HR = - sum(m_h_mole .* m_wdot[1:ns, :], dims=1)'
    # @test m_cp_mass ≈ f.cp_mass rtol = 1.e-4
    # @test m_cp_mole ≈ f.cp_mole rtol = 1.e-4
    # @test m_λ_mix ≈ f.thermal_conductivity rtol = 1.e-4
    # @test m_wdot[1:end - 1, :] ≈ f.net_production_rates rtol = 1.e-4
    # @test HR ≈ f.heat_release_rate rtol = 1.e-4

    return vcat(F...)
end

cal_wdot = Wdot(f.P)
p = zeros(nr)
z = f.grid
ng = length(z)
ny = ns + 2
yall = hcat(f.Y', f.T, f.velocity)
yv = reshape(yall, 1, ng * ny)
yL = vcat(f.Y[:, 1], f.T[1])
ind_f = findfirst(f.T .> (f.T[1] + f.T[end]) / 2.0)
T_f = f.T[ind_f]

Fv = residual(gas, cal_wdot, p, z, yv, yL, ind_f, T_f)

@time Fy = ForwardDiff.jacobian(yv -> residual(gas, cal_wdot, p, z, yv, yL, ind_f, T_f), yv)
@time Fp = ForwardDiff.jacobian(p -> residual(gas, cal_wdot, p, z, yv, yL, ind_f, T_f), p)

dydp = Fy \ Fp

sens = dydp[ns + 2, :]