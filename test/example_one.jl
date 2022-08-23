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

    soln_newton = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:nr)
    soln_lm_kyf = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:lm_kyf)
    soln_lm_ar  = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:lm_ar)
    soln_trust  = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:tr)
    soln_dogleg = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:dogleg)
    soln_nk     = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:nk)
    soln_jfnk   = nlboxsolve(gershwin,x0,l,u,xtol=1e-15,ftol=1e-15,method=:jfnk)

    soln_newton_sparse = nlboxsolve(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15,method=:nr,sparsejac=:yes)
    soln_lm_kyf_sparse = nlboxsolve(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15,method=:lm_kyf,sparsejac=:yes)
    soln_lm_ar_sparse  = nlboxsolve(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15,method=:lm_ar,sparsejac=:yes)
    soln_trust_sparse  = nlboxsolve(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15,method=:tr,sparsejac=:yes)
    soln_dogleg_sparse = nlboxsolve(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15,method=:dogleg,sparsejac=:yes)
    soln_nk_sparse     = nlboxsolve(gershwin,x0,l,u,ftol=1e-15,xtol=1e-15,method=:nk,sparsejac=:yes)

    test_one   = maximum(abs,soln_lm_kyf.zero - soln_newton.zero) < 1e-14
    test_two   = maximum(abs,soln_lm_ar.zero  - soln_newton.zero) < 1e-14
    test_three = maximum(abs,soln_trust.zero  - soln_newton.zero) < 1e-14
    test_four  = maximum(abs,soln_dogleg.zero - soln_newton.zero) < 1e-14
    test_five  = maximum(abs,soln_nk.zero     - soln_newton.zero) < 1e-14
    test_six   = maximum(abs,soln_jfnk.zero   - soln_newton.zero) < 1e-14

    test_seven  = maximum(abs,soln_newton_sparse.zero - soln_newton.zero) < 1e-14
    test_eight  = maximum(abs,soln_lm_kyf_sparse.zero - soln_newton.zero) < 1e-14
    test_nine   = maximum(abs,soln_lm_ar_sparse.zero  - soln_newton.zero) < 1e-14
    test_ten    = maximum(abs,soln_trust_sparse.zero  - soln_newton.zero) < 1e-14
    test_eleven = maximum(abs,soln_dogleg_sparse.zero - soln_newton.zero) < 1e-14
    test_twelve = maximum(abs,soln_nk_sparse.zero     - soln_newton.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve

end

function test_nlboxsolve_inplace()

    function gershwin!(f,x)

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

    soln_newton = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:nr)
    soln_lm_kyf = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:lm_kyf)
    soln_lm_ar  = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:lm_ar)
    soln_trust  = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:tr)
    soln_dogleg = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:dogleg)
    soln_nk     = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:nk)
    soln_jfnk   = nlboxsolve(gershwin!,x0,l,u,xtol=1e-15,ftol=1e-15,method=:jfnk)

    soln_newton_sparse = nlboxsolve(gershwin!,x0,l,u,ftol=1e-15,xtol=1e-15,method=:nr,sparsejac=:yes)
    soln_lm_kyf_sparse = nlboxsolve(gershwin!,x0,l,u,ftol=1e-15,xtol=1e-15,method=:lm_kyf,sparsejac=:yes)
    soln_lm_ar_sparse  = nlboxsolve(gershwin!,x0,l,u,ftol=1e-15,xtol=1e-15,method=:lm_ar,sparsejac=:yes)
    soln_trust_sparse  = nlboxsolve(gershwin!,x0,l,u,ftol=1e-15,xtol=1e-15,method=:tr,sparsejac=:yes)
    soln_dogleg_sparse = nlboxsolve(gershwin!,x0,l,u,ftol=1e-15,xtol=1e-15,method=:dogleg,sparsejac=:yes)
    soln_nk_sparse     = nlboxsolve(gershwin!,x0,l,u,ftol=1e-15,xtol=1e-15,method=:nk,sparsejac=:yes)

    test_one   = maximum(abs,soln_lm_kyf.zero - soln_newton.zero) < 1e-14
    test_two   = maximum(abs,soln_lm_ar.zero  - soln_newton.zero) < 1e-14
    test_three = maximum(abs,soln_trust.zero  - soln_newton.zero) < 1e-14
    test_four  = maximum(abs,soln_dogleg.zero - soln_newton.zero) < 1e-14
    test_five  = maximum(abs,soln_nk.zero     - soln_newton.zero) < 1e-14
    test_six   = maximum(abs,soln_jfnk.zero   - soln_newton.zero) < 1e-14

    test_seven  = maximum(abs,soln_newton_sparse.zero - soln_newton.zero) < 1e-14
    test_eight  = maximum(abs,soln_lm_kyf_sparse.zero - soln_newton.zero) < 1e-14
    test_nine   = maximum(abs,soln_lm_ar_sparse.zero  - soln_newton.zero) < 1e-14
    test_ten    = maximum(abs,soln_trust_sparse.zero  - soln_newton.zero) < 1e-14
    test_eleven = maximum(abs,soln_dogleg_sparse.zero - soln_newton.zero) < 1e-14
    test_twelve = maximum(abs,soln_nk_sparse.zero     - soln_newton.zero) < 1e-14

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve

end

function test_mcpsolve_outplace()

    function test(x::Array{T,1}) where {T<:Number}

        f = Array{T,1}(undef,length(x))
    
        f[1] = x[1]^3 - 8
        f[2] = x[2] - x[3] + x[2]^3 + 3
        f[3] = x[2] + x[3] + 2*x[3]^3 - 3
        f[4] = x[4] + 2*x[4]^3
    
        return f
    
    end
    
    # Solve system

    x0 = [0.5,0.5,0.5,0.5]
    l  = [-1.0,-1.0,-1.0,-1.0]
    u  = [1.0,1.0,1.0,1.0]

    soln_newton_mid = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:nr)
    soln_lm_kyf_mid = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:lm_kyf)
    soln_lm_ar_mid  = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:lm_ar)
    soln_trust_mid  = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:tr)
    soln_dogleg_mid = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:dogleg)
    soln_nk_mid     = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:nk)
    soln_jfnk_mid   = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:mid,method=:jfnk)

    soln_newton_mid_sparse = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:mid,method=:nr,sparsejac=:yes)
    soln_lm_kyf_mid_sparse = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:mid,method=:lm_kyf,sparsejac=:yes)
    soln_lm_ar_mid_sparse  = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:mid,method=:lm_ar,sparsejac=:yes)
    soln_trust_mid_sparse  = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:mid,method=:tr,sparsejac=:yes)
    soln_dogleg_mid_sparse = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:mid,method=:dogleg,sparsejac=:yes)
    soln_nk_mid_sparse     = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:mid,method=:nk,sparsejac=:yes)

    soln_newton_fb = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:nr)
    soln_lm_kyf_fb = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:lm_kyf)
    soln_lm_ar_fb  = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:lm_ar)
    #soln_trust_fb  = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:tr)
    soln_dogleg_fb = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:dogleg)
    soln_nk_fb     = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:nk)
    soln_jfnk_fb   = mcpsolve(test,x0,l,u,xtol=1e-8,ftol=1e-8,reformulation=:fb,method=:jfnk)

    soln_newton_fb_sparse = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:fb,method=:nr,sparsejac=:yes)
    soln_lm_kyf_fb_sparse = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:fb,method=:lm_kyf,sparsejac=:yes)
    soln_lm_ar_fb_sparse  = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:fb,method=:lm_ar,sparsejac=:yes)
    #soln_trust_fb_sparse  = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:fb,method=:tr,sparsejac=:yes)
    soln_dogleg_fb_sparse = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:fb,method=:dogleg,sparsejac=:yes)
    soln_nk_fb_sparse     = mcpsolve(test,x0,l,u,ftol=1e-8,xtol=1e-8,reformulation=:fb,method=:nk,sparsejac=:yes)

    test_one   = maximum(abs,soln_lm_kyf_mid.zero - soln_newton_mid.zero) < 1e-7
    test_two   = maximum(abs,soln_lm_ar_mid.zero  - soln_newton_mid.zero) < 1e-7
    test_three = maximum(abs,soln_trust_mid.zero  - soln_newton_mid.zero) < 1e-7
    test_four  = maximum(abs,soln_dogleg_mid.zero - soln_newton_mid.zero) < 1e-7
    test_five  = maximum(abs,soln_nk_mid.zero     - soln_newton_mid.zero) < 1e-7
    test_six   = maximum(abs,soln_jfnk_mid.zero   - soln_newton_mid.zero) < 1e-7

    test_seven  = maximum(abs,soln_newton_mid_sparse.zero - soln_newton_mid.zero) < 1e-7
    test_eight  = maximum(abs,soln_lm_kyf_mid_sparse.zero - soln_newton_mid.zero) < 1e-7
    test_nine   = maximum(abs,soln_lm_ar_mid_sparse.zero  - soln_newton_mid.zero) < 1e-7
    test_ten    = maximum(abs,soln_trust_mid_sparse.zero  - soln_newton_mid.zero) < 1e-7
    test_eleven = maximum(abs,soln_dogleg_mid_sparse.zero - soln_newton_mid.zero) < 1e-7
    test_twelve = maximum(abs,soln_nk_mid_sparse.zero     - soln_newton_mid.zero) < 1e-7

    test_thirteen  = maximum(abs,soln_lm_kyf_fb.zero - soln_lm_ar_fb.zero) < 1e-7
    test_fourteen  = maximum(abs,soln_lm_ar_fb.zero  - soln_lm_ar_fb.zero) < 1e-7
    #test_fifteen   = maximum(abs,soln_trust_fb.zero  - soln_lm_ar_fb.zero) < 1e-7
    test_sixteen   = maximum(abs,soln_dogleg_fb.zero - soln_lm_ar_fb.zero) < 1e-7
    test_seventeen = maximum(abs,soln_nk_fb.zero     - soln_lm_ar_fb.zero) < 1e-7
    test_eighteen  = maximum(abs,soln_jfnk_fb.zero   - soln_lm_ar_fb.zero) < 1e-7

    test_nineteen     = maximum(abs,soln_newton_fb_sparse.zero - soln_lm_ar_fb.zero) < 1e-7
    test_twenty       = maximum(abs,soln_lm_kyf_fb_sparse.zero - soln_lm_ar_fb.zero) < 1e-7
    test_twenty_one   = maximum(abs,soln_lm_ar_fb_sparse.zero  - soln_lm_ar_fb.zero) < 1e-7
    #test_twenty_two   = maximum(abs,soln_trust_fb_sparse.zero  - soln_lm_ar_fb.zero) < 1e-7
    test_twenty_three = maximum(abs,soln_dogleg_fb_sparse.zero - soln_lm_ar_fb.zero) < 1e-7
    test_twenty_four  = maximum(abs,soln_nk_fb_sparse.zero     - soln_lm_ar_fb.zero) < 1e-7

    return test_one, test_two, test_three, test_four, test_five, test_six, test_seven, test_eight, test_nine, test_ten, test_eleven,
           test_twelve, test_thirteen, test_fourteen, #=test_fifteen,=# test_sixteen, test_seventeen, test_eighteen, test_nineteen,
           test_twenty, test_twenty_one, #=test_twenty_two,=# test_twenty_three, test_twenty_four

end

test_nlboxsolve_outplace()
test_nlboxsolve_inplace()
test_mcpsolve_outplace()
