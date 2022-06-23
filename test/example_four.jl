# This code presents an example to illustrate how NLboxsolve.jl can be used

using NLboxsolve

function test_nlboxsolve_outplace()

    function test6(x)

        f = similar(x)
    
        for i in eachindex(x)
            f[i] = cos(x[i]) - 1.0
        end
    
        return f
    
    end
                   
    # Solve test6 model

    n = 20
    x0 = ones(n)*0.5
    l = ones(n)*-0.5
    u = ones(n)*0.7
    
    soln_newton     = constrained_newton(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm         = constrained_levenberg_marquardt(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf     = constrained_levenberg_marquardt_kyf(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar      = constrained_levenberg_marquardt_ar(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg     = constrained_dogleg_solver(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_bmp = constrained_dogleg_bmp_solver(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk         = constrained_newton_krylov(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_jfnk       = constrained_jacobian_free_newton_krylov(test6,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln_newton_sparse     = constrained_newton_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_sparse         = constrained_levenberg_marquardt_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf_sparse     = constrained_levenberg_marquardt_kyf_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar_sparse      = constrained_levenberg_marquardt_ar_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_sparse     = constrained_dogleg_solver_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_bmp_sparse = constrained_dogleg_bmp_solver_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_sparse         = constrained_newton_krylov_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln2 = nlboxsolve(test6,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one   = maximum(abs,soln_newton.zero       - soln2.zero) < 1e-8
    test_two   = maximum(abs,soln_lm.zero           - soln2.zero) < 1e-8
    test_three = maximum(abs,soln_lm_kyf.zero       - soln2.zero) < 1e-8
    test_four  = maximum(abs,soln_lm_ar.zero        - soln2.zero) < 1e-8
    test_five  = maximum(abs,soln_dogleg.zero       - soln2.zero) < 1e-8
    test_six   = maximum(abs,soln_dogleg_bmp.zero   - soln2.zero) < 1e-8
    test_seven = maximum(abs,soln_nk.zero           - soln2.zero) < 1e-8
    test_eight = maximum(abs,soln_jfnk.zero         - soln2.zero) < 1e-8

    test_nine     = maximum(abs,soln_newton_sparse.zero       - soln2.zero) < 1e-8
    test_ten      = maximum(abs,soln_lm_sparse.zero           - soln2.zero) < 1e-8
    test_eleven   = maximum(abs,soln_lm_kyf_sparse.zero       - soln2.zero) < 1e-8
    test_twelve   = maximum(abs,soln_lm_ar_sparse.zero        - soln2.zero) < 1e-8
    test_thirteen = maximum(abs,soln_dogleg_sparse.zero       - soln2.zero) < 1e-8
    test_fourteen = maximum(abs,soln_dogleg_bmp_sparse.zero   - soln2.zero) < 1e-8
    test_fifteen  = maximum(abs,soln_nk_sparse.zero           - soln2.zero) < 1e-8

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, test_fifteen

end

function test_nlboxsolve_inplace()

    function test6(f,x)

        for i in eachindex(x)
            f[i] = cos(x[i]) - 1.0
        end
    
    end
                   
    # Solve test6 model

    n = 20
    x0 = ones(n)*0.5
    l = ones(n)*-0.5
    u = ones(n)*0.7
    
    soln_newton     = constrained_newton(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm         = constrained_levenberg_marquardt(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf     = constrained_levenberg_marquardt_kyf(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar      = constrained_levenberg_marquardt_ar(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg     = constrained_dogleg_solver(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_bmp = constrained_dogleg_bmp_solver(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk         = constrained_newton_krylov(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_jfnk       = constrained_jacobian_free_newton_krylov(test6,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln_newton_sparse     = constrained_newton_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_sparse         = constrained_levenberg_marquardt_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf_sparse     = constrained_levenberg_marquardt_kyf_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar_sparse      = constrained_levenberg_marquardt_ar_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_sparse     = constrained_dogleg_solver_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg_bmp_sparse = constrained_dogleg_bmp_solver_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_sparse         = constrained_newton_krylov_sparse(test6,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln2 = nlboxsolve(test6,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one   = maximum(abs,soln_newton.zero       - soln2.zero) < 1e-8
    test_two   = maximum(abs,soln_lm.zero           - soln2.zero) < 1e-8
    test_three = maximum(abs,soln_lm_kyf.zero       - soln2.zero) < 1e-8
    test_four  = maximum(abs,soln_lm_ar.zero        - soln2.zero) < 1e-8
    test_five  = maximum(abs,soln_dogleg.zero       - soln2.zero) < 1e-8
    test_six   = maximum(abs,soln_dogleg_bmp.zero   - soln2.zero) < 1e-8
    test_seven = maximum(abs,soln_nk.zero           - soln2.zero) < 1e-8
    test_eight = maximum(abs,soln_jfnk.zero         - soln2.zero) < 1e-8

    test_nine     = maximum(abs,soln_newton_sparse.zero       - soln2.zero) < 1e-8
    test_ten      = maximum(abs,soln_lm_sparse.zero           - soln2.zero) < 1e-8
    test_eleven   = maximum(abs,soln_lm_kyf_sparse.zero       - soln2.zero) < 1e-8
    test_twelve   = maximum(abs,soln_lm_ar_sparse.zero        - soln2.zero) < 1e-8
    test_thirteen = maximum(abs,soln_dogleg_sparse.zero       - soln2.zero) < 1e-8
    test_fourteen = maximum(abs,soln_dogleg_bmp_sparse.zero   - soln2.zero) < 1e-8
    test_fifteen  = maximum(abs,soln_nk_sparse.zero           - soln2.zero) < 1e-8

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, test_fifteen

end

test_nlboxsolve_outplace()
test_nlboxsolve_inplace()
