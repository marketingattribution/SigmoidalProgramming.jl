using SigmoidalProgramming
using Dates, DataFrames, Statistics

export generate_event_curves


function generate_event_curves(
    media_parameters,
    event,
    event_data,
    causal,
    maxiters=10000,
    verbose=0,
    n_segments::Int=20,
    lower_budget::Float64=0.1,
    upper_budget::Float64=2.0
)
    retention = exp.(log(0.5) ./ (media_parameters[!, :half_life]))

    curves = DataFrame()
    flightings = DataFrame()

    for i=1:size(media_parameters)[1]
        println("variable: ", media_parameters[i,:variable])
        product_id = media_parameters[i, :product_id]
        event_id = event[event[!, :variable].==media_parameters[i, :variable], :id]

        periods = causal[((causal[!, :product_id].==product_id)
                .& (causal[!, :variable].=="core")), [:period]
        ]

        baseline = causal[((causal[!, :product_id].==product_id)
                .& (causal[!, :variable].=="core")), [:causal]
        ]
        baseline = convert(Array, baseline)
        baseline = vcat(baseline, baseline)
        cost = convert(Array, event_data[(event_data[!, :event_id].==event_id), [:cost_per_point]])
        spend = sum(event_data[(event_data[!, :event_id].==event_id), :spend])

        output_curve = event_flighting(
            periods,
            baseline,
            cost,
            media_parameters[i,:scale],
            media_parameters[i,:shape],
            media_parameters[i,:coefficient],
            retention[i],
            spend,
            maxiters=10000,
            verbose=0,
            n_segments=n_segments,
            lower_budget=lower_budget,
            upper_budget=upper_budget
        )
        output_curve[!, :variable] .= media_parameters[i,:variable]
        flightings = vcat(flightings, output_curve)
        curve = combine(groupby(output_curve, [:variable, :spend, :pct, :status]), :lb .=> mean)
        curves = vcat(curves, curve)
    end
    curves = rename(curves, :lb_mean => :revenue)
    flightings = rename(flightings, :lb => :revenue)

    return curves, flightings
end


"to find the optimal flighting for a single event"
function event_flighting(
    periods::DataFrame,
    baseline::Array{Float64},
    cost::Array{Float64},
    scale::Float64,
    shape::Float64,
    coef::Float64,
    retention::Float64,
    spend::Float64;
    nweeks::Int=52,
    lower_budget::Float64=0.5,
    upper_budget::Float64=1.5,
    l=Nothing,
    u=Nothing,
    n_segments::Int=20,
    maxiters::Int64,
    verbose::Int=0,
    TOL::Float64=0.01
)

    if l == Nothing
        l = fill(0, 3 * nweeks)
    end
    if u == Nothing
        u = cat(fill(spend/minimum(cost)*6, 2 * nweeks), fill(spend / minimum(cost), nweeks), dims=(1,))
    end

    # inflection point
    if shape > 1
        z = cat(fill(scale*((shape - 1) / shape) ^ (1 / shape), nweeks * 2), fill(0, nweeks), dims=(1,))
    else
        z = fill(0, nweeks * 3)
    end

    # budget constraint matrix
    A = hcat(zeros(1, nweeks * 2), transpose(cost))

    # adstock-grps constraints
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

    # functions
    fs1 = Function[x -> weibull(x, coef, scale, shape) * baseline[i] for i=1 : nweeks * 2]
    dfs1 = Function[x -> weibull_prime(x, coef, scale, shape) * baseline[i] for i=1 : nweeks * 2]

    fs2 = Function[x -> 0 for i=1 : nweeks]
    dfs2 = Function[x -> 0 for i=1 : nweeks]
    fs = vcat(fs1, fs2)
    dfs = vcat(dfs1, dfs2)

    output_curve = DataFrame()

    # loop through all the budget levels
    for B = range(
        spend * lower_budget,
        spend * upper_budget,
        step = max(spend * (upper_budget - lower_budget) / (n_segments-1), 1E-10)
    )
        println("budget: ", B)
        println(now())

        problem = LinearSP(fs, dfs, z, A, [B], C, D)

        # find initial point
        grps = find_flat_weekly_pattern(
                retention,
                scale,
                shape,
                B / (sum(cost)/length(cost))
            )
        max_kpi, optim_flighting, adstock_grps = find_max_kpi_flat_fighting(
            baseline,
            cost,
            grps,
            retention,
            scale,
            shape,
            coef
        )

        # branch and bound
        pq, bestnodes, lbs, ubs, status = @time solve_sp(
            l, u, problem, adstock_grps; TOL=TOL, maxiters=maxiters, verbose=verbose,
            maxiters_noimprovement = 1000
        )

        if length(lbs) > 0
            grps=bestnodes[end].x[nweeks * 2 + 1: nweeks * 3]
            lb=lbs[end]

            diff_in_out = (adstock_grps[nweeks * 2 + 1 : nweeks * 3] - grps) ./ grps
            se = replace!(diff_in_out .* diff_in_out, Inf=>NaN)
            mse = mean(se[isnan.(se).==false])
        else
            mse = 0.0
            lb = 0.0
        end
        if mse <= 0.00001 && status == 0
            pq2, bestnodes2, lbs2, ubs2, status2 = @time solve_sp(
                l, u, problem, Nothing; TOL=TOL, maxiters=maxiters, verbose=verbose,
                maxiters_noimprovement = 1000
            )
            if length(lbs2) > 0
                lb2=lbs2[end]
            else
                lb2 = 0.0
            end
            if lb2 > lb
                pq = pq2
                bestnodes = bestnodes2
                lbs = lbs2
                status = status2
            end
        end

        if length(lbs) > 0
            grps = DataFrame(
                spend=fill(B, nweeks),
                grps=bestnodes[end].x[nweeks * 2 + 1: nweeks * 3],
                lb=fill(lbs[end], nweeks),
                status = fill(status, nweeks),
                pct = fill(B / spend * 100, nweeks)
            )

            output_curve = vcat(output_curve, hcat(periods, grps))
        end
    end

    println(now())
    return output_curve
end


function find_flat_weekly_pattern(retention::Float64, scale::Float64, shape::Float64, total_grps::Float64, nweeks::Int=52)
    if shape<=1.0
        grps_maintenance_week = total_grps / (nweeks - 2 + 2 / (1 - retention))
        grps_first_week = grps_maintenance_week / (1 - retention)
        weekly_grps = vcat(grps_first_week, fill(grps_maintenance_week, nweeks - 2), grps_first_week)
    else
        f = x->weibull(x, 1.0, scale, shape)
        df = x->weibull_prime(x, 1.0, scale, shape)
        z = scale*((shape - 1) / shape) ^ (1 / shape)
        grps_first_week = find_w(f, df, 0, total_grps, z)
        grps_maintenance_week = grps_first_week * (1 - retention)
        n_pos_weeks = min(Int(ceil((total_grps - grps_first_week) / grps_maintenance_week)), nweeks - 1)
        grps_leftover = total_grps - grps_first_week - n_pos_weeks * grps_maintenance_week
        if n_pos_weeks > 0
            if grps_leftover > 0
                adjustment = total_grps / (total_grps - grps_leftover)
                grps_first_week = grps_first_week * adjustment
                grps_maintenance_week = grps_maintenance_week * adjustment
                weekly_grps = vcat(grps_first_week, fill(grps_maintenance_week, n_pos_weeks))
            else
                weekly_grps = vcat(
                    grps_first_week,
                    fill(grps_maintenance_week, n_pos_weeks - 1),
                    grps_maintenance_week + grps_leftover
                )
            end
        else
            return [grps_first_week]
        end
    end

    return weekly_grps
end


function find_max_kpi_flat_fighting(baseline, cost, grps, retention, scale, shape, coefficient, nweeks=52)
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

        kpi = sum(((1 .- exp.(-((adstock / scale) .^ shape))) * coefficient) .* baseline)

        if kpi > max_kpi
            max_kpi = kpi
            optim_flighting = test_grps
            adstock_grps = vcat(adstock, test_grps)
        end
    end

    return max_kpi, optim_flighting, adstock_grps
end
