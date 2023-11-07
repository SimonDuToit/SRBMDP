using StatsBase
mutable struct PFTFilter{PM<:POMDP,RNG<:AbstractRNG} <: Updater
    pomdp::PM
    rng::RNG
    #p::PMEM # initial and post-resampling particles (p → pp → p)
    n_p::Int
    n_k::Int
    wr::Float64 #weighted return
end

function PFTFilter(pomdp::POMDP, n_p::Int, n_k::Int, rng::AbstractRNG)
    S = statetype(pomdp)
    return PFTFilter(
        pomdp,
        rng,
        n_p,
        n_k,
        0.0,
        )
end

PFTFilter(pomdp::POMDP, n_p::Int, n_k::Int) = PFTFilter(pomdp, n_p, n_k, Random.default_rng())

function initialize_belief(pf::PFTFilter, source)
    N = pf.n_p
    w = Vector{Float64}(undef, N)
    p = collect(rand(source) for i in 1:N)
    fill!(w, inv(N))
    return PFTBelief(p, w, pf.pomdp, rand(initialobs(pf.pomdp, rand(initialstate(pf.pomdp)))))
end

"""
predict!
    - propagate b(up.p) → up.p
reweight!
    - update up.w
    - s ∈ b(up.p), sp ∈ up.p
resample!
    - resample up.p → b
"""
function update(up::PFTFilter, b::PFTBelief, a, o)
    N = n_particles(b)
    weighted_return = 0.0
    b.t += 1

    #bp_particles, bp_weights = gen_empty_belief(planner.cache, N)
    T = typeof(rand(initialstate(up.pomdp)))
    bp_particles, bp_weights = Vector{T}(undef, N), Vector{Float64}(undef, N)

    for (i,(s,w)) in enumerate(weighted_particles(b))
        # Sampling
        if !isterminal(up.pomdp, s)
            #(sp, r) = sr_gen(planner.obs_req, rng, pomdp, s, a) # @gen(:sp,:r)(pomdp, s, a, rng)
            (sp, r) = @gen(:sp,:r)(up.pomdp, s, a, up.rng)
        else
            (sp,r) = (s, 0.0)
        end

        # Reweighting
        @inbounds begin
            bp_particles[i] = sp
            w = weight(b, i)
            bp_weights[i] = w*pdf(POMDPs.observation(up.pomdp, s, a, sp), o)
        end

        weighted_return += r*w
    end

    if !all(iszero, bp_weights)
        normalize!(bp_weights, 1)
    else
        fill!(bp_weights, inv(N))
    end

    bp = PFTBelief(bp_particles, bp_weights, up.pomdp, o)
    up.wr = weighted_return

    #println(bp)

    # resampling
    Neff = inv(sum(ParticleFilters.weights(bp).^2))
    if true
        new_ps = sample(particles(bp), StatsBase.Weights(ParticleFilters.weights(bp)), N)
        copyto!(bp.particles, new_ps)
        fill!(bp.weights, inv(N))
        #end
    end

    #println(bp.weights)

    # OG resampling (not mine)
    if false
        ps = Vector{T}(undef, N)
        ws = 1.0
        r = rand(up.rng)/N
        c = bp_weights[1]
        i = 1
        U = r

        for m in 1:N
            while U > c && i < N
                i += 1
                c += bp.weights[i]
            end
            U += ws/N
            @inbounds ps[m] = bp.particles[i]
        end

        copyto!(bp.particles, ps)
        fill!(bp.weights, inv(N))
    end
    return bp
end
