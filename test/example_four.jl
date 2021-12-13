# This code presents an example to illustrate how NLboxsolve.jl can be used

using NLboxsolve

function test_nlboxsolve()

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
    
    soln_newton = constrained_newton(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm     = constrained_levenberg_marquardt(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_kyf = constrained_levenberg_marquardt_kyf(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_fan = constrained_levenberg_marquardt_fan(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_lm_ar  = constrained_levenberg_marquardt_ar(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_dogleg = constrained_dogleg_solver(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk     = constrained_newton_krylov(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_nk_fs  = constrained_newton_krylov_fs(test6,x0,l,u,ftol=1e-15,xtol=1e-15)
    soln_jfnk   = constrained_jacobian_free_newton_krylov(test6,x0,l,u,ftol=1e-15,xtol=1e-15)

    soln1 = nlboxsolve(test6,x0,l,u,xtol=1e-15,ftol=1e-15)

    test_one   = maximum(abs,soln_newton.zero - soln1.zero) < 1e-14
    test_two   = maximum(abs,soln_lm.zero     - soln1.zero) < 1e-14
    test_three = maximum(abs,soln_lm_kyf.zero - soln1.zero) < 1e-14
    test_four  = maximum(abs,soln_lm_fan.zero - soln1.zero) < 1e-14
    test_five  = maximum(abs,soln_lm_ar.zero  - soln1.zero) < 1e-14
    test_six   = maximum(abs,soln_dogleg.zero - soln1.zero) < 1e-14
    test_seven = maximum(abs,soln_nk.zero     - soln1.zero) < 1e-14
    test_eight = maximum(abs,soln_nk_fs.zero  - soln1.zero) < 1e-14
    test_nine  = maximum(abs,soln_jfnk.zero   - soln1.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine

end

test_nlboxsolve()
