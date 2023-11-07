#import Base.eltype

mutable struct PFTBelief{T} <: AbstractParticleBelief{T}
    particles::Vector{T}
    weights::Vector{Float64}
    #non_terminal_ws::Float64
    t::Int64
    o
end

function PFTBelief(particles::Vector{T}, weights::Vector{Float64}, pomdp::POMDP, o) where {T}
    #terminal_ws = 0.0
    #for (s,w) in zip(particles, weights)
    #    !isterminal(pomdp, s) && (terminal_ws += w)
    #end
    #return PFTBelief{T}(particles, weights, terminal_ws)
    return PFTBelief{T}(particles, weights, 1, o)
end

@inline ParticleFilters.n_particles(b::PFTBelief) = length(b.particles)
@inline ParticleFilters.particles(p::PFTBelief) = p.particles
ParticleFilters.weighted_particles(b::PFTBelief) = (
    b.particles[i]=>b.weights[i] for i in 1:length(b.particles)
)
@inline ParticleFilters.weight_sum(b::PFTBelief) = 1.0
@inline ParticleFilters.weight(b::PFTBelief, i::Int) = b.weights[i]
@inline ParticleFilters.particle(b::PFTBelief, i::Int) = b.particles[i]
@inline ParticleFilters.weights(b::PFTBelief) = b.weights

function Random.rand(rng::AbstractRNG, b::PFTBelief)
    t = rand(rng)
    i = 1
    cw = b.weights[1]
    while cw < t && i < length(b.weights)
        i += 1
        @inbounds cw += weight(b,i)
    end
    return particle(b,i)
end

function non_terminal_sample(rng::AbstractRNG, pomdp::POMDP, b::PFTBelief)
    t = rand(rng)*b.non_terminal_ws
    i = 1
    cw = isterminal(pomdp,particle(b,1)) ? 0.0 : weight(b,1)
    while cw < t && i < length(b.weights)
        i += 1
        isterminal(pomdp,particle(b,i)) && continue
        @inbounds cw += weight(b,i)
    end
    return i
end

Random.rand(b::PFTBelief) = Random.rand(Random.GLOBAL_RNG, b)

function convert_s_old(::Type{A}, b::PFTBelief, bmdp) where A<:AbstractArray
    #s = Array{eltype(A)}(undef, 0)
    s = Array{Float32}(undef, 0)
    for i in 1:n_particles(b)
        append!(s, convert_s(Array{Float32}, particle(b, i), bmdp.pomdp))
        append!(s, weight(b, i))
    end
    return s
end

function convert_s_old(::Type{AbstractParticleBelief{S}}, vec::V, bmdp::Union{MDP{AbstractParticleBelief{S}}, POMDP{AbstractParticleBelief{S}}}) where {S,V<:AbstractArray}
    #lol = kek
    Nv = size(vec)[1]
    pomdp_state = rand(initialstate(bmdp.pomdp))
    Ni = size(convert_s(Array{Float32}, pomdp_state, bmdp.pomdp))[1]
    Np = Int(Nv / (Ni+1))
    particles, weights = Vector{typeof(pomdp_state)}(undef, Np), Vector{Float64}(undef, Np)
    index = 1
    for pi in 1:Np
        p = convert_s(typeof(pomdp_state), vec[index:index+Ni-1], bmdp.pomdp)
        w = vec[index+Ni]
        particles[pi] = p
        weights[pi] = w
        index += Ni+1
    end
    
    return PFTBelief(particles, weights, bmdp.pomdp)
end

function convert_s(::Type{A}, b::PFTBelief, bmdp) where A<:AbstractArray
    e = sum(p * log(p) for p in values(probdict(b)) if p > 0.0)
    #s = convert_s(A, rand(b), bmdp.pomdp)
    s = convert_s(Array{Float32}, rand(b), bmdp.pomdp)
    append!(s, e)
    return s
end

function convert_s_max(::Type{A}, b::PFTBelief, bmdp) where A<:AbstractArray
    max_i = argmax(b.weights)
    m_s = particle(b, max_i)
    s = convert_s(Array{Float32}, m_s, bmdp.pomdp)
    m_w = weight(b, max_i)
    append!(s, m_w)
    return s
end

function convert_s_dummy(::Type{S}, vec::V, bmdp) where {S,V<:AbstractArray}
    v = vec[1:size(vec)[1]-1]
    s = convert_s(S.parameters[1], v, bmdp.pomdp)
    N = 100
    particles = fill(s, N)
    weights = fill(inv(N), N)
    return PFTBelief(particles, weights, bmdp.pomdp)
end

function convert_s_many(::Type{A}, b::PFTBelief, bmdp) where A<:AbstractArray
    e = sum(p * log(p) for p in values(probdict(b)) if p > 0.0)
    #s = convert_s(A, rand(b), bmdp.pomdp)
    s = convert_s(Array{Float32}, rand(b), bmdp.pomdp)
    for i in 2:10
        si = convert_s(Array{Float32}, rand(b), bmdp.pomdp)
        append!(s, si)
    end
    append!(s, e)
    return s
end

function convert_s_perdim(::Type{A}, b::PFTBelief, bmdp) where A<:AbstractArray
    probdicts = probdict_per_dim(b, bmdp)
    es = Array{Float32}(undef, size(probdicts)[1])
    for (i, dict) in enumerate(probdicts)
        es[i] = sum(w * log(w) for w in values(dict) if w > 0.0)
    end
    #s = convert_s(A, rand(b), bmdp.pomdp)
    s = convert_s(Array{Float32}, rand(b), bmdp.pomdp)
    append!(s, es)
    return s
end
