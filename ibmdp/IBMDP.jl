module IBMDP

using POMDPs
# Simon
import POMDPs: initialstate, initialize_belief, update, gen, actions, isterminal, discount, convert_s
using Random
using POMDPTools # action_info
#import POMDPTools: discount
#using LinearAlgebra # normalize!
#using PushVectors
using ParticleFilters

# Simon
using Distributions
using LinearAlgebra

# Simon
using CUDA
using Flux
import Flux: gpu

include("belief.jl")
include("PFTFilter.jl")

export GenerativeInformationBMDP
export PFTFilter

"""
    GenerativeInformationBMDP(pomdp, updater)

Create a generative model of the belief MDP corresponding to POMDP `pomdp` with belief updates performed by `updater`.
"""
struct GenerativeInformationBMDP{P<:POMDP, U<:Updater, B, A} <: MDP{B, A}
    pomdp::P
    updater::U
    λ::Float64
    squared_IG::Bool
end

function GenerativeInformationBMDP(pomdp::P, up::U, λ::Float64, squared_IG::Bool) where {P<:POMDP, U<:Updater}
    # XXX hack to determine belief type
    #b0 = initialize_belief(up, initialstate(pomdp))
    T = typeof(rand(initialstate(pomdp)))
    return GenerativeInformationBMDP{P, U, AbstractParticleBelief{T}, actiontype(pomdp)}(pomdp, up, λ, squared_IG,)
end

function gen(bmdp::GenerativeInformationBMDP, b::AbstractParticleBelief{S}, a, rng::AbstractRNG) where S
    #s = rand(rng, b)
    #if isterminal(bmdp.pomdp, s)
    #    bp = gbmdp_handle_terminal(bmdp.pomdp, bmdp.updater, b, s, a, rng::AbstractRNG)::typeof(b)
    #    return (sp=bp, r=0.0)
    #end
    #sp, o, r = @gen(:sp, :o, :r)(bmdp.pomdp, s, a, rng) # maybe this should have been generate_or?
    #bp = update(bmdp.updater, b, a, o)
    #return (sp=bp, r=r)

    # start
    #p_idx = non_terminal_sample(rng, bmdp.pomdp, b)
    #p_idx = non_terminal_sample(rng, bmdp.pomdp, b)
    #sample_s = particle(b, p_idx)
    sample_s = rand(b)
    o, r = @gen(:o, :r)(bmdp.pomdp, sample_s, a, rng)
    bp = update(bmdp.updater, b, a, o)
    if true
        weighted_return = bmdp.updater.wr
    else
        weighted_return = r
    end
    #planner.sol.resample && resample!(planner, bp)
    

    # Simon
    # note to self - multiplying by small constant 
    # wrecks things for rock sample? Investigate.
    #λ = 100.0*0.5^b.t
    #λ = 10.0
    #println(λ)
    #println(b.weights)
    #println(bp.weights)
    #println("===========")
    #println(length(probdict(b)))
    info_gain = discrete_gain(bmdp.pomdp, b, bp)
    #println(info_gain)
    #info_gain = max(0, info_gain)
    if bmdp.squared_IG  
        weighted_return = weighted_return + bmdp.λ*sign(info_gain)*info_gain^2
    else
        weighted_return = weighted_return + bmdp.λ*info_gain
    end

    # return bp::PFTBelief{S}, o::O, weighted_return::Float64
    return (sp=bp, r=weighted_return)
    #return(sp=convert_i(bp, bmdp), r=weighted_return)
end

actions(bmdp::GenerativeInformationBMDP{P,B,A}, b::B) where {P,B,A} = actions(bmdp.pomdp, b)
actions(bmdp::GenerativeInformationBMDP) = actions(bmdp.pomdp)

isterminal(bmdp::GenerativeInformationBMDP, b) = all(isterminal(bmdp.pomdp, s) for s in support(b))

discount(bmdp::GenerativeInformationBMDP) = discount(bmdp.pomdp)

# override this if you want to handle it in a special way
function gbmdp_handle_terminal(pomdp::POMDP, updater::Updater, b::AbstractParticleBelief{S}, s, a, rng) where {S}
    @warn("""
         Sampled a terminal state for a GenerativeInformationBMDP transition - not sure how to proceed, but will try.

         See $(@__FILE__) and implement a new method of POMDPToolbox.gbmdp_handle_terminal if you want special behavior in this case.

         """, maxlog=1)
    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
    bp = update(updater, b, a, o)
    return bp
end

function initialstate(bmdp::GenerativeInformationBMDP)
    return Deterministic(initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)))
    #s = convert_i(initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)), bmdp)
    #return Deterministic(s)
end

function discrete_gain(pomdp::POMDP, b::AbstractParticleBelief{S}, bp::AbstractParticleBelief{S}) where {S}
    if isterminal(pomdp, particle(bp, 1))
        return 0.0
    else
        ibp = sum(p * log(p) for p in values(probdict(bp)) if p > 0.0)
        ib = sum(p * log(p) for p in values(probdict(b)) if p > 0.0)
        return ibp - ib
    end
end

function probdict(b::AbstractParticleBelief{S}) where {S}
    probs = Dict{S, Float64}()
    for (i,p) in enumerate(particles(b))
        if haskey(probs, p)
            probs[p] += weight(b, i)/weight_sum(b)
        else
            probs[p] = weight(b, i)/weight_sum(b)
        end
    end
    return probs
end

function probdict_per_dim(b::AbstractParticleBelief{S}, bmdp) where {S}
    #println(b)
    M = size(convert_s(Array{Float32}, rand(b), bmdp.pomdp))[1]
    dicts = Array{Dict{Float32, Float32}}(undef,M)
    for i in 1:size(dicts)[1]
        dicts[i] = Dict{Float32, Float32}()
    end
    #probs = Dict{S, Float64}()
    for (i,particle) in enumerate(particles(b))
        p = convert_s(Array{Float32}, particle, bmdp.pomdp)
        for (j,dict) in enumerate(dicts)
            if haskey(dict, p[j])
                dict[p[j]] += weight(b, i)/weight_sum(b)
            else
                dict[p[j]] = weight(b, i)/weight_sum(b)
            end
        end
    end
    #println(dicts)
    #println("================")
    return dicts
end
end