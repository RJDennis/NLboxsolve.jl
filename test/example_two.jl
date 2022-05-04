# This code presents an example to illustrate how NLboxsolve.jl can be used

using NLboxsolve

function test_nlboxsolve_outplace()

    function test2(x)

        f = similar(x)
    
        f[1] = x[1]*x[3] + x[4]*x[1] + x[4]*x[3] - 0.0
        f[2] = x[2]*x[3] + x[4]*x[2] + x[4]*x[3] - 0.0
        f[3] = x[1]*x[2] + x[1]*x[3] + x[2]*x[3] - 1.0
        f[4] = x[1]*x[2] + x[4]*x[1] + x[4]*x[2] - 0.0
    
        return f
    end
    
    # Solve test2 model

    x0 = [0.5, 0.5, 0.5, -0.3]
    l = [-Inf,-Inf,-Inf,-Inf]
    u = [Inf,Inf,Inf,Inf]

    soln_newton = constrained_newton(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm     = constrained_levenberg_marquardt(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf = constrained_levenberg_marquardt_kyf(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan = constrained_levenberg_marquardt_fan(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar  = constrained_levenberg_marquardt_ar(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg = constrained_dogleg_solver(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk     = constrained_newton_krylov(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs  = constrained_newton_krylov_fs(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_jfnk   = constrained_jacobian_free_newton_krylov(test2,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln_newton_sparse = constrained_newton_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_sparse     = constrained_levenberg_marquardt_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf_sparse = constrained_levenberg_marquardt_kyf_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan_sparse = constrained_levenberg_marquardt_fan_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar_sparse  = constrained_levenberg_marquardt_ar_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_sparse = constrained_dogleg_solver_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_sparse     = constrained_newton_krylov_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs_sparse  = constrained_newton_krylov_fs_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln1 = nlboxsolve(test2,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one   = maximum(abs,soln_newton.zero - soln1.zero) < 1e-14
    test_two   = maximum(abs,soln_lm.zero     - soln1.zero) < 1e-14
    test_three = maximum(abs,soln_lm_kyf.zero - soln1.zero) < 1e-14
    test_four  = maximum(abs,soln_lm_fan.zero - soln1.zero) < 1e-14
    test_five  = maximum(abs,soln_lm_ar.zero  - soln1.zero) < 1e-14
    test_six   = maximum(abs,soln_dogleg.zero - soln1.zero) < 1e-14
    test_seven = maximum(abs,soln_nk.zero     - soln1.zero) < 1e-14
    test_eight = maximum(abs,soln_nk_fs.zero  - soln1.zero) < 1e-14
    test_nine  = maximum(abs,soln_jfnk.zero   - soln1.zero) < 1e-14

    test_ten       = maximum(abs,soln_newton_sparse.zero - soln1.zero) < 1e-14
    test_eleven    = maximum(abs,soln_lm_sparse.zero     - soln1.zero) < 1e-14
    test_twelve    = maximum(abs,soln_lm_kyf_sparse.zero - soln1.zero) < 1e-14
    test_thirteen  = maximum(abs,soln_lm_fan_sparse.zero - soln1.zero) < 1e-14
    test_fourteen  = maximum(abs,soln_lm_ar_sparse.zero  - soln1.zero) < 1e-14
    test_fifteen   = maximum(abs,soln_dogleg_sparse.zero - soln1.zero) < 1e-14
    test_sixteen   = maximum(abs,soln_nk_sparse.zero     - soln1.zero) < 1e-14
    test_seventeen = maximum(abs,soln_nk_fs_sparse.zero  - soln1.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, test_fifteen, test_sixteen, test_seventeen

end

function test_nlboxsolve_inplace()

    function test2(f,x)

        f[1] = x[1]*x[3] + x[4]*x[1] + x[4]*x[3] - 0.0
        f[2] = x[2]*x[3] + x[4]*x[2] + x[4]*x[3] - 0.0
        f[3] = x[1]*x[2] + x[1]*x[3] + x[2]*x[3] - 1.0
        f[4] = x[1]*x[2] + x[4]*x[1] + x[4]*x[2] - 0.0
    
    end
    
    # Solve test2 model

    x0 = [0.5, 0.5, 0.5, -0.3]
    l = [-Inf,-Inf,-Inf,-Inf]
    u = [Inf,Inf,Inf,Inf]

    soln_newton = constrained_newton(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm     = constrained_levenberg_marquardt(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf = constrained_levenberg_marquardt_kyf(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan = constrained_levenberg_marquardt_fan(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar  = constrained_levenberg_marquardt_ar(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg = constrained_dogleg_solver(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk     = constrained_newton_krylov(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs  = constrained_newton_krylov_fs(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_jfnk   = constrained_jacobian_free_newton_krylov(test2,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln_newton_sparse = constrained_newton_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_sparse     = constrained_levenberg_marquardt_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf_sparse = constrained_levenberg_marquardt_kyf_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan_sparse = constrained_levenberg_marquardt_fan_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar_sparse  = constrained_levenberg_marquardt_ar_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_sparse = constrained_dogleg_solver_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_sparse     = constrained_newton_krylov_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs_sparse  = constrained_newton_krylov_fs_sparse(test2,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln1 = nlboxsolve(test2,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one   = maximum(abs,soln_newton.zero - soln1.zero) < 1e-14
    test_two   = maximum(abs,soln_lm.zero     - soln1.zero) < 1e-14
    test_three = maximum(abs,soln_lm_kyf.zero - soln1.zero) < 1e-14
    test_four  = maximum(abs,soln_lm_fan.zero - soln1.zero) < 1e-14
    test_five  = maximum(abs,soln_lm_ar.zero  - soln1.zero) < 1e-14
    test_six   = maximum(abs,soln_dogleg.zero - soln1.zero) < 1e-14
    test_seven = maximum(abs,soln_nk.zero     - soln1.zero) < 1e-14
    test_eight = maximum(abs,soln_nk_fs.zero  - soln1.zero) < 1e-14
    test_nine  = maximum(abs,soln_jfnk.zero   - soln1.zero) < 1e-14

    test_ten       = maximum(abs,soln_newton_sparse.zero - soln1.zero) < 1e-14
    test_eleven    = maximum(abs,soln_lm_sparse.zero     - soln1.zero) < 1e-14
    test_twelve    = maximum(abs,soln_lm_kyf_sparse.zero - soln1.zero) < 1e-14
    test_thirteen  = maximum(abs,soln_lm_fan_sparse.zero - soln1.zero) < 1e-14
    test_fourteen  = maximum(abs,soln_lm_ar_sparse.zero  - soln1.zero) < 1e-14
    test_fifteen   = maximum(abs,soln_dogleg_sparse.zero - soln1.zero) < 1e-14
    test_sixteen   = maximum(abs,soln_nk_sparse.zero     - soln1.zero) < 1e-14
    test_seventeen = maximum(abs,soln_nk_fs_sparse.zero  - soln1.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, test_fifteen, test_sixteen, test_seventeen

end

test_nlboxsolve_outplace()
test_nlboxsolve_inplace()
