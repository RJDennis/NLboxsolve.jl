# This code presents an example to illustrate how NLboxsolve.jl can be used

using NLboxsolve

function test_nlboxsolve_outplace()

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

    soln_newton     = constrained_newton(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_newton_ms  = constrained_newton_ms(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm         = constrained_levenberg_marquardt(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_kyf     = constrained_levenberg_marquardt_kyf(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_fan     = constrained_levenberg_marquardt_fan(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_ar      = constrained_levenberg_marquardt_ar(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_tr         = constrained_trust_region(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_dogleg     = constrained_dogleg_solver(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_dogleg_bmp = constrained_dogleg_bmp_solver(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_nk         = constrained_newton_krylov(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_nk_fs      = constrained_newton_krylov_fs(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_jfnk       = constrained_jacobian_free_newton_krylov(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)

    soln_newton_sparse     = constrained_newton_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_newton_ms_sparse  = constrained_newton_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_sparse         = constrained_levenberg_marquardt_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf_sparse     = constrained_levenberg_marquardt_kyf_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan_sparse     = constrained_levenberg_marquardt_fan_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar_sparse      = constrained_levenberg_marquardt_ar_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_tr_sparse         = constrained_dogleg_solver_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_sparse     = constrained_dogleg_solver_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_bmp_sparse = constrained_dogleg_solver_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_sparse         = constrained_newton_krylov_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs_sparse      = constrained_newton_krylov_fs_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln1 = nlboxsolve(gershwin,x0,l,u,method = :dogleg,xtol=1e-15,ftol=1e-15)
    soln2 = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one    = maximum(abs,soln_newton.zero       - soln2.zero) < 1e-14
    test_two    = maximum(abs,soln_newton_ms.zero    - soln2.zero) < 1e-14
    test_three  = maximum(abs,soln_lm.zero           - soln2.zero) < 1e-14
    test_four   = maximum(abs,soln_lm_kyf.zero       - soln2.zero) < 1e-14
    test_five   = maximum(abs,soln_lm_fan.zero       - soln2.zero) < 1e-14
    test_six    = maximum(abs,soln_lm_ar.zero        - soln2.zero) < 1e-14
    #test_seven  = maximum(abs,soln_tr.zero           - soln2.zero) < 1e-14
    test_eight  = maximum(abs,soln_dogleg.zero       - soln2.zero) < 1e-14
    test_nine   = maximum(abs,soln_dogleg_bmp.zero   - soln2.zero) < 1e-14
    test_ten    = maximum(abs,soln_nk.zero           - soln2.zero) < 1e-14
    test_eleven = maximum(abs,soln_nk_fs.zero        - soln2.zero) < 1e-14
    test_twelve = maximum(abs,soln_jfnk.zero         - soln2.zero) < 1e-14

    test_thirteen     = maximum(abs,soln_newton_sparse.zero       - soln2.zero) < 1e-14
    test_fourteen     = maximum(abs,soln_newton_ms_sparse.zero    - soln2.zero) < 1e-14
    test_fifteen      = maximum(abs,soln_lm_sparse.zero           - soln2.zero) < 1e-14
    test_sixteen      = maximum(abs,soln_lm_kyf_sparse.zero       - soln2.zero) < 1e-14
    test_seventeen    = maximum(abs,soln_lm_fan_sparse.zero       - soln2.zero) < 1e-14
    test_eighteen     = maximum(abs,soln_lm_ar_sparse.zero        - soln2.zero) < 1e-14
    test_nineteen     = maximum(abs,soln_tr_sparse.zero           - soln2.zero) < 1e-14
    test_twenty       = maximum(abs,soln_dogleg_sparse.zero       - soln2.zero) < 1e-14
    test_twenty_one   = maximum(abs,soln_dogleg_bmp_sparse.zero   - soln2.zero) < 1e-14
    test_twenty_two   = maximum(abs,soln_nk_sparse.zero           - soln2.zero) < 1e-14
    test_twenty_three = maximum(abs,soln_nk_fs_sparse.zero        - soln2.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, test_fifteen, test_sixteen, test_seventeen, test_eighteen, test_nineteen, test_twenty,
           test_twenty_one, test_twenty_two, test_twenty_three

end

function test_nlboxsolve_inplace()

    function gershwin(f,x)

        f[1] = 2*x[1] - x[5]
        f[2] = 2*x[2] - x[5]
        f[3] = 2*x[3] - x[5]
        f[4] = 2*x[4] - x[5] + x[6]
        f[5] = x[1] + x[2] + x[3] + x[4] - 1.0
        f[6] = x[6]*(x[4]-0.2)

    end

    # Solve Gershwin system

    x0 = [0.25,0.25,0.45,0.05,0.1,0.1]
    l = [-Inf,-Inf,-Inf,-Inf,0.0,0.0]
    u = [Inf,Inf,Inf,0.2,Inf,Inf]

    soln_newton     = constrained_newton(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_newton_ms  = constrained_newton_ms(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm         = constrained_levenberg_marquardt(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_kyf     = constrained_levenberg_marquardt_kyf(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_fan     = constrained_levenberg_marquardt_fan(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_lm_ar      = constrained_levenberg_marquardt_ar(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_tr         = constrained_trust_region(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_dogleg     = constrained_dogleg_solver(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_dogleg_bmp = constrained_dogleg_bmp_solver(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_nk         = constrained_newton_krylov(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_nk_fs      = constrained_newton_krylov_fs(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)
    soln_jfnk       = constrained_jacobian_free_newton_krylov(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)

    soln_newton_sparse     = constrained_newton_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_newton_ms_sparse  = constrained_newton_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_sparse         = constrained_levenberg_marquardt_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf_sparse     = constrained_levenberg_marquardt_kyf_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan_sparse     = constrained_levenberg_marquardt_fan_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar_sparse      = constrained_levenberg_marquardt_ar_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_tr_sparse         = constrained_dogleg_solver_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_sparse     = constrained_dogleg_solver_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_bmp_sparse = constrained_dogleg_solver_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_sparse         = constrained_newton_krylov_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs_sparse      = constrained_newton_krylov_fs_sparse(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln1 = nlboxsolve(gershwin,x0,l,u,method = :dogleg,xtol=1e-15,ftol=1e-15)
    soln2 = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one    = maximum(abs,soln_newton.zero       - soln2.zero) < 1e-14
    test_two    = maximum(abs,soln_newton_ms.zero    - soln2.zero) < 1e-14
    test_three  = maximum(abs,soln_lm.zero           - soln2.zero) < 1e-14
    test_four   = maximum(abs,soln_lm_kyf.zero       - soln2.zero) < 1e-14
    test_five   = maximum(abs,soln_lm_fan.zero       - soln2.zero) < 1e-14
    test_six    = maximum(abs,soln_lm_ar.zero        - soln2.zero) < 1e-14
    #test_seven  = maximum(abs,soln_tr.zero           - soln2.zero) < 1e-14
    test_eight  = maximum(abs,soln_dogleg.zero       - soln2.zero) < 1e-14
    test_nine   = maximum(abs,soln_dogleg_bmp.zero   - soln2.zero) < 1e-14
    test_ten    = maximum(abs,soln_nk.zero           - soln2.zero) < 1e-14
    test_eleven = maximum(abs,soln_nk_fs.zero        - soln2.zero) < 1e-14
    test_twelve = maximum(abs,soln_jfnk.zero         - soln2.zero) < 1e-14

    test_thirteen     = maximum(abs,soln_newton_sparse.zero       - soln2.zero) < 1e-14
    test_fourteen     = maximum(abs,soln_newton_ms_sparse.zero    - soln2.zero) < 1e-14
    test_fifteen      = maximum(abs,soln_lm_sparse.zero           - soln2.zero) < 1e-14
    test_sixteen      = maximum(abs,soln_lm_kyf_sparse.zero       - soln2.zero) < 1e-14
    test_seventeen    = maximum(abs,soln_lm_fan_sparse.zero       - soln2.zero) < 1e-14
    test_eighteen     = maximum(abs,soln_lm_ar_sparse.zero        - soln2.zero) < 1e-14
    test_nineteen     = maximum(abs,soln_tr_sparse.zero           - soln2.zero) < 1e-14
    test_twenty       = maximum(abs,soln_dogleg_sparse.zero       - soln2.zero) < 1e-14
    test_twenty_one   = maximum(abs,soln_dogleg_bmp_sparse.zero   - soln2.zero) < 1e-14
    test_twenty_two   = maximum(abs,soln_nk_sparse.zero           - soln2.zero) < 1e-14
    test_twenty_three = maximum(abs,soln_nk_fs_sparse.zero        - soln2.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, test_fifteen, test_sixteen, test_seventeen, test_eighteen, test_nineteen, test_twenty,
           test_twenty_one, test_twenty_two, test_twenty_three

end

test_nlboxsolve_outplace()
test_nlboxsolve_inplace()
