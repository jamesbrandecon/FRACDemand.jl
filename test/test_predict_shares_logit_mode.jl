using Test
using DataFrames
using FRACDemand

@testset "predict_shares! logit mode (nonlinear empty)" begin
    df = DataFrame(
        market_ids = [1, 1, 2, 2],
        product_ids = [1, 2, 1, 2],
        prices = [1.0, 2.0, 1.5, 0.5],
        x = [0.0, 1.0, -1.0, 2.0],
        # Needed for define_problem() LHS construction; keep outside share positive.
        shares = [0.20, 0.10, 0.15, 0.05],
        # Needed for make_vars() instrument check (regex: r"demand_instruments").
        demand_instruments_1 = [0.1, 0.2, 0.3, 0.4],
    )

    prob = define_problem(
        data = df,
        linear = ["prices", "x"],
        nonlinear = String[],
        fixed_effects = String[],
        constrained = false,
        se_type = "simple",
    )

    βp = -1.25
    βx = 0.4
    prob.estimated_parameters = Dict(
        :β_prices => βp,
        :β_x => βx,
    )

    predict_shares!(df, prob)

    for m in unique(df.market_ids)
        idx = findall(df.market_ids .== m)
        deltas = βp .* df.prices[idx] .+ βx .* df.x[idx]
        denom = 1 + sum(exp.(deltas))
        expected = exp.(deltas) ./ denom
        @test all(isapprox.(df.shares[idx], expected; atol=1e-10, rtol=1e-10))
    end
end
