using SigmoidalProgramming
using Dates, DataFrames

export event_flighting


"to find the optimal flighting for a single event"
function event_flighting(
    baseline::Array{Float64},
    cost::Array{Float64},
    scale::Float64,
    shape::Float64,
    coef::Float64,
    retention::Float64,
    spend::Float64;
    maxiters::Int64,
    verbose::Int=0,
    TOL::Float64=0.01,
    nweeks::Int=52,
    nconstr::Int=1,
    lower_budget::Float64=0.5,
    upper_budget::Float64=1.5,
    n_segments::Int=20,
)

    l = fill(0, 3 * nweeks)
    u = cat(fill(spend, 2 * nweeks), fill(spend / 5, nweeks), dims=(1,))
    A = hcat(zeros(nconstr, nweeks * 2), ones(nconstr, nweeks))

    z = cat(fill(scale*((shape - 1) / shape) ^ (1 / shape), nweeks * 2), fill(0, nweeks), dims=(1,))

    zz = zeros(nweeks * 3)
    C = copy(zz)
    C[1] = 1
    C[105] = -1
    C = transpose(C)
    D = fill(0, nweeks * 2)

    for i = 2 : nweeks
        c = copy(zz)
        c[i-1] = -retention
        c[i] = 1
        c[i + nweeks * 2] = -1
        C = vcat(C, transpose(c))
    end

    for i = nweeks + 1 : nweeks * 2
        c = copy(zz)
        c[i - 1] = -retention
        c[i] = 1
        C = vcat(C, transpose(c))
    end

    fs1 = Function[x -> weibull(x, coef, scale, shape) * baseline[i] for i=1 : nweeks * 2]
    fs2 = Function[x -> -x * cost[i] for i=1 : nweeks]
    fs = vcat(fs1, fs2)
    dfs1 = Function[x -> weibull_prime(x, coef, scale, shape) * baseline[i] for i=1 : nweeks * 2]
    dfs2 = Function[x -> -cost[i] for i=1 : nweeks]
    dfs = vcat(dfs1, dfs2)

    output_curve = DataFrame()
    iterlog = DataFrame()

    for B = range(
        spend * lower_budget,
        spend * upper_budget,
        step=spend * (upper_budget - lower_budget) / n_segments
    )
        println("budget: ", B)
        println(now())

        problem = LinearSP(fs, dfs, z, A, [B], C, D)

        l = fill(0, 3 * nweeks)
        u = cat(fill(spend, 2 * nweeks), fill(spend / 2, nweeks), dims=(1,))

        # find initial point
        grps = find_weekly_pattern(
                retention,
                scale,
                shape,
                B
            )
        max_kpi, optim_flighting, adstock_grps = find_max_kpi_fighting(
            baseline,
            cost,
            grps,
            retention,
            scale,
            shape,
            coef
        )

        # branch and bound
        pq, bestnodes, lbs, ubs, log = @time solve_sp(
            l, u, problem; TOL=TOL, maxiters=maxiters, verbose=verbose, init_x=adstock_grps
        )

        grps = DataFrame(
            period = 1 : nweeks,
            spend=fill(B, nweeks),
            grps=bestnodes[end].x[nweeks * 2 + 1: nweeks * 3],
            lb=fill(lbs[end], nweeks)
        )
        output_curve = vcat(output_curve, grps)
        log[!, :budget] .= B
        iterlog = vcat(iterlog, log)
    end

    println(now())
    return iterlog, output_curve
end


function find_weekly_pattern(retention::Float64, scale::Float64, shape::Float64, budget::Float64, nweeks::Int=52)
    if round(shape, digits=1)<=1.0
        grps_maintenance_week = budget / (nweeks - 2 + 2 / (1 - retention))
        grps_first_week = grps_maintenance_week / (1 - retention)
        weekly_grps = vcat(grps_first_week, fill(grps_maintenance_week, nweeks - 2), grps_first_week)
    else
        f = x->weibull(x, 1.0, scale, shape)
        df = x->weibull_prime(x, 1.0, scale, shape)
        z = scale*((shape - 1) / shape) ^ (1 / shape)
        grps_first_week = find_w(f, df, 0, budget, z)
        grps_maintenance_week = grps_first_week * (1 - retention)
        n_pos_weeks = Int(floor((budget - grps_first_week) / grps_maintenance_week))
        grps_last_week = budget - grps_first_week - n_pos_weeks * grps_maintenance_week
        weekly_grps = vcat(grps_first_week, fill(grps_maintenance_week, n_pos_weeks), grps_last_week)
    end

    return weekly_grps
end


function find_max_kpi_fighting(baseline, cost, grps, retention, scale, shape, coefficient, nweeks=52)
    n_grps_weeks = size(grps)[1]
    max_kpi = -1E10
    optim_flighting = fill(0, n_grps_weeks)
    adstock_grps = fill(0, n_grps_weeks * 3)

    for w = 1 : nweeks - n_grps_weeks + 1
        adstock = zeros(nweeks * 2)
        test_grps = zeros(nweeks)
        for i = w : nweeks * 2
            g = i - w + 1
            if g == 1
                adstock[i] = grps[g]
                test_grps[i] = grps[g]
            elseif g <= n_grps_weeks
                adstock[i] = adstock[i-1] * retention + grps[g]
                test_grps[i] = grps[g]
            else
                adstock[i] = adstock[i-1] * retention
            end
        end
        kpi = sum(((1 .- exp.(-((adstock / scale) .^ shape))) * coefficient) .* baseline) - sum(test_grps .* cost)
        if kpi > max_kpi
            max_kpi = kpi
            optim_flighting = test_grps
            adstock_grps = vcat(adstock, test_grps)
        end
    end

    return max_kpi, optim_flighting, adstock_grps
end
