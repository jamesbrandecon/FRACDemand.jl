using FRACDemand, Random, DataFrames, Plots
using Statistics

# Simulation settings
nsim = 1000
J1, J2, T, B = 20, 20, 500, 1
β = [-2.0, 1.5]
Σ = [0.1 0.0; 0.0 0.1]
ξ_var = 0.5

# storage for all elasticity pairs
all_elast = DataFrame(mse=Float64[], mape=Float64[], 
    betax=Float64[], betap=Float64[], σ2x=Float64[], σ2p=Float64[], σcov=Float64[])

all_elast_nocov = DataFrame(mse=Float64[], mape=Float64[],
    betax=Float64[], betap=Float64[], σ2x=Float64[], σ2p=Float64[], σcov=Float64[])

for s in 1:nsim
    Random.seed!(s)
    # simulate and reshape
    df = FRACDemand.sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
    df[df.market_ids .>= T,"product_ids"] .+=  J1-2
    sort!(df, [:market_ids, :product_ids])
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2
    original_xi = df.xi
    original_delta = df.xi .+ df.prices * β[1] + df.x * β[2] 
    df[!,"xi"] .= original_xi
    
    scatter(
        problem.data.shares, df.shares
    )
    # set up IV and estimate
    problem = define_problem(
      data          = df,
      linear        = ["prices","x"],
      nonlinear     = ["prices","x"],
      cov           = [("prices","x")],
      fixed_effects = ["product_ids", "market_ids"],
      se_type       = "bootstrap",
      constrained   = false
    )
    estimate!(problem)

    # true elasticities
    truth = FRACDemand.sim_true_price_elasticities(df, β, Σ)

    # estimated elasticities
    est_elast = DataFrame(market_ids=Int[], product_i=Int[], product_j=Int[], elasticity=Float64[])
    for subdf in groupby(problem.data, :market_ids)
        p, x, xi = subdf.prices, subdf.x, subdf.xi
        fe = "market_FEs" in names(subdf) ? first(subdf.market_FEs) : 0
        βhat = [problem.estimated_parameters[:β_prices], problem.estimated_parameters[:β_x]]
        Ehat = FRACDemand.sim_price_elasticities(p, x, xi, βhat, Σ; market_FE=fe)
        mid, J = first(subdf.market_ids), length(p)
        for i in 1:J, j in 1:J
            push!(est_elast, (market_ids=mid, product_i=i, product_j=j, elasticity=Ehat[i,j]))
        end
    end

    # collect data for sim
    # DataFrame(truth=truth.elasticity, est=est_elast.elasticity)
    append!(all_elast, DataFrame(
        mse  = mean((truth.elasticity .- est_elast.elasticity).^2), 
        mape = mean(abs.(truth.elasticity .- est_elast.elasticity) ./ abs.(truth.elasticity)), 
        betax = problem.estimated_parameters[:β_x], 
        betap = problem.estimated_parameters[:β_prices],
        σ2x  = problem.estimated_parameters[:σ2_x],
        σ2p  = problem.estimated_parameters[:σ2_prices],
        σcov = problem.estimated_parameters[:σcov_prices_x]
    ))

    # set up IV and estimate
    problem_nocov = define_problem(
      data          = select(df, Not(:demand_instruments3)),
      linear        = ["prices","x"],
      nonlinear     = ["prices","x"],
      fixed_effects = ["product_ids"],
      se_type       = "bootstrap",
      constrained   = false
    )
    estimate!(problem_nocov)

    # estimated elasticities
    est_elast = DataFrame(market_ids=Int[], product_i=Int[], product_j=Int[], elasticity=Float64[])
    for subdf in groupby(problem_nocov.data, :market_ids)
        p, x, xi = subdf.prices, subdf.x, subdf.xi
        fe = "market_FEs" in names(subdf) ? first(subdf.market_FEs) : 0
        βhat = [problem_nocov.estimated_parameters[:β_prices], problem_nocov.estimated_parameters[:β_x]]
        Ehat = FRACDemand.sim_price_elasticities(p, x, xi, βhat, Σ; market_FE=fe)
        mid, J = first(subdf.market_ids), length(p)
        for i in 1:J, j in 1:J
            push!(est_elast, (market_ids=mid, product_i=i, product_j=j, elasticity=Ehat[i,j]))
        end
    end

    # collect data for sim
    append!(all_elast_nocov, DataFrame(
        mse  = mean((truth.elasticity .- est_elast.elasticity).^2), 
        mape = mean(abs.(truth.elasticity .- est_elast.elasticity) ./ abs.(truth.elasticity)), 
        betax = problem_nocov.estimated_parameters[:β_x], 
        betap = problem_nocov.estimated_parameters[:β_prices],
        σ2x  = problem_nocov.estimated_parameters[:σ2_x],
        σ2p  = problem_nocov.estimated_parameters[:σ2_prices],
        σcov = 0; 
    ))

    # track progress
    print(".")
end

function cov_plots(without_cov, with_cov)
    # Plot for σ²_x
    p1 = histogram(
        with_cov.σ2x,
        bins=30,
        label="With Cov",
        xlabel="σ²_x",
        ylabel="Frequency",
        title="Estimated σ²_x",
        color=:skyblue, # Assign a color for this histogram
        legend=:topright
    )
    histogram!(p1,
        without_cov.σ2x,
        bins=30,
        label="Without Cov",
        color=:lightcoral, # Assign a different color
        alpha=0.5
    )
    vline!(p1, [Σ[2,2]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Plot for σ²_prices
    p2 = histogram(
        with_cov.σ2p,
        bins=30,
        label="With Cov",
        xlabel="σ²_prices",
        ylabel="Frequency",
        title="Estimated σ²_prices",
        color=:skyblue, # Assign a different color
        legend=:topright
    )
     histogram!(p2,
        without_cov.σ2p,
        bins=30,
        label="Without Cov",
        color=:lightcoral, # Assign a different color
        alpha=0.5
    )
    vline!(p2, [Σ[1,1]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Plot for σ_cov(prices, x)
    p3 = histogram(
        with_cov.σcov,
        bins=30,
        label="With Cov",
        xlabel="σ_cov(prices, x)",
        ylabel="Frequency",
        title="Estimated σ_cov(prices, x)",
        color=:skyblue, # Assign another different color
        legend=:topright
    )
    vline!(p3,
        [0],
        bins=30,
        label="Without Cov",
        color=:lightcoral, # Assign a different color
        alpha=1.0, 
        linestyle=:solid
    )
    vline!(p3, [Σ[1,2]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Combine plots into a single figure with a layout and an overall title
    p_out = plot(p1, p2, p3,
        layout = (3, 1),          # Arrange plots vertically
        size = (700, 900)        # Adjust figure size as needed
        # plot_title = "Distribution of Estimated Var/Covar Params" # Overall title
        )
    return p_out
end

function beta_plots(without_cov, with_cov)
    p1 = histogram(
        with_cov.betax,
        bins=30,
        label="With Cov",
        xlabel="β_x",
        ylabel="Frequency",
        title="Estimated β_x",
        color=:skyblue, # Assign a color for this histogram
        legend=:topright
    )
    histogram!(p1,
        without_cov.betax,
        bins=30,
        label="Without Cov",
        color=:lightcoral, # Assign a different color
        alpha=0.5
    )
    vline!(p1, [β[2]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Plot for σ²_prices
    p2 = histogram(
        with_cov.betap,
        bins=30,
        label="With Cov",
        xlabel="β_prices",
        ylabel="Frequency",
        title="Estimated β_prices",
        color=:skyblue, # Assign a different color
        legend=:topright
    )
    histogram!(p2,
        without_cov.betap,
        bins=30,
        label="Without Cov",
        color=:lightcoral, # Assign a different color
        alpha=0.5
    )
    vline!(p2, [β[1]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Combine plots into a single figure with a layout and an overall title
    p_out = plot(p1, p2,
        layout = (2, 1),          # Arrange plots vertically
        size = (700, 900)        # Adjust figure size as needed
        # plot_title = "Distribution of Estimated Var/Covar Params" # Overall title
        )
    return p_out
end

function accuracy_plots(without_cov, with_cov)

    p1 = histogram(
        all_elast.mape;
        bins=30, xlabel="MAPE", ylabel="Count",
        title="Mean Absolute Percentage Error of Price Elasticities", 
        label="With Cov Estiamtion", 
        normalize = :density, 
        alpha = 0.5,
        color = :skyblue
        )
    histogram!(
        p1,
        all_elast_nocov.mape;
        bins=30, 
        xlabel="MAPE", ylabel="Count",
        label="Without Cov", 
        normalize = :density, 
        alpha = 0.5, 
        color = :lightcoral
        )

    p2 = histogram(
        all_elast.mse;
        bins=30, xlabel="MSE", ylabel="Count",
        title="Mean Squared Error of Price Elasticities", 
        label="With Cov", 
        normalize = :density, 
        alpha = 0.5,
        color = :skyblue
        )
    histogram!(
        p2,
        all_elast_nocov.mse;
        bins=30, 
        xlabel="MSE", ylabel="Count",
        label="Without Cov Estimation", 
        normalize = :density, 
        alpha = 0.5, 
        color = :lightcoral
        )
    
    return plot(p1, p2,
        layout = (2, 1),          # Arrange plots vertically
        size = (700, 900)        # Adjust figure size as needed
        # plot_title = "Distribution of Estimated Var/Covar Params" # Overall title
        )
end

# -------------------------------------
# Make plots 
# -------------------------------------
cov_plots(all_elast_nocov, all_elast)

beta_plots(all_elast_nocov, all_elast)

accuracy_plots(all_elast_nocov, all_elast)

scatter(
    all_elast_nocov.mape, all_elast.mape;
    xlabel="No cov", ylabel="w/ Cov",
    title="MAPE of Price Elasticities", legend=false, alpha=0.2
    )
plot!(
    all_elast_nocov.mape, all_elast_nocov.mape;
    color=:red, linestyle=:dash, linewidth=2,
    label="45 degree line"
    )
scatter(
    all_elast_nocov.mse, all_elast.mse;
    xlabel="No cov", ylabel="w/ Cov",
    title="MSE of Price Elasticities", legend=false, alpha=0.2
    )
plot!(
    all_elast_nocov.mse, all_elast_nocov.mse;
    color=:red, linestyle=:dash, linewidth=2,
    label="45 degree line"
    )