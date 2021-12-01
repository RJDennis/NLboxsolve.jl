# This code presents an example to illustrate how NLboxsolve.jl can be used

using NLboxsolve

function test_nlboxsolve()

    function gershwin(x)

        f = similar(x)

        f[1] = 2*x[1] - x[5]
        f[2] = 2*x[2] - x[5]
        f[3] = 2*x[3] - x[5]
        f[4] = 2*x[4] - x[5] + x[6]
        f[5] = x[1] + x[2] + x[3] + x[4] - 1.0
        f[6] = x[6]*(x[4]-0.2)

        return f

    end

    # Solve Gershwin system

    x0 = [0.25,0.25,0.45,0.05,0.1,0.1]
    l = [-Inf,-Inf,-Inf,-Inf,0.0,0.0]
    u = [Inf,Inf,Inf,0.2,Inf,Inf]

    soln_newton = constrained_newton(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm     = constrained_levenberg_marquardt(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_kyf = constrained_levenberg_marquardt_kyf(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_fan = constrained_levenberg_marquardt_fan(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_ar  = constrained_levenberg_marquardt_ar(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_dogleg = constrained_dogleg_solver(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_nk     = constrained_newton_krylov(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_nk_fs  = constrained_newton_krylov_fs(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)

    soln1 = nlboxsolve(gershwin,x0,l,u,method = :dogleg,xtol=1e-15,ftol=1e-15)
    soln2 = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one   = maximum(abs,soln_newton.zero - soln2.zero) < 1e-14
    test_two   = maximum(abs,soln_lm.zero     - soln2.zero) < 1e-14
    test_three = maximum(abs,soln_lm_kyf.zero - soln2.zero) < 1e-14
    test_four  = maximum(abs,soln_lm_fan.zero - soln2.zero) < 1e-14
    test_five  = maximum(abs,soln_lm_ar.zero  - soln2.zero) < 1e-14
    test_six   = maximum(abs,soln_dogleg.zero - soln2.zero) < 1e-14
    test_seven = maximum(abs,soln_nk.zero     - soln2.zero) < 1e-14
    test_eight = maximum(abs,soln_nk_fs.zero  - soln2.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight

end

test_nlboxsolve()
