using AutoGrad: Param, @diff, value, grad

using LinearAlgebra: I, norm


Identity(i;scale=1) = zeros(i,i) + I * scale

sigm(x) = 1 / (1 + exp(-x))
tanh(x) = (begin exp2x = exp(2*x) ; (exp2x-1) / (exp2x+1) end)
relu(x) = max(0, x)

softmax(x) = (begin expx = exp.(x) ; expx ./ sum(expx) end)

cross_entropy(label, prediction) = -(label .* log.(prediction))
binary_cross_entropy(label, prediction) = -(label .* log.(prediction) + (1 .- label) .* log.(1 .- prediction))
mse(label, prediction) = (label - prediction) .^2

xavier(weight; gain=5/3) = (rand(size(weight)) .* 2 .- 1) .* gain .* sqrt(6/sum(size(weight)))


mutable struct Recurrent
    wk::Param
    # bk::Param
    wf::Param
    # bf::Param
    wn::Param
    # bn::Param
    wo::Param
    # bo::Param
    Recurrent(in_size,layer_size) =
    begin
        wk1 = randn(in_size,layer_size)
        wk2 = randn(layer_size,layer_size)
        # bk  = zeros(1,layer_size)
        wf1 = randn(in_size,layer_size)
        wf2 = randn(layer_size,layer_size)
        # bf  = zeros(1,layer_size)
        wn1 = randn(in_size,layer_size)
        wn2 = randn(layer_size,layer_size)
        # bn  = zeros(1,layer_size)
        wo1 = randn(in_size,layer_size)
        wo2 = randn(layer_size,layer_size)
        # bo  = zeros(1,layer_size)
    new(
        Param(vcat(wk1,wk2)),
        # Param(bk),
        Param(vcat(wf1,wf2)),
        # Param(bf),
        Param(vcat(wn1,wn2)),
        # Param(bn),
        Param(vcat(wo1,wo2)),
        # Param(bo),
    )
    end
end

(layer::Recurrent)(state, in) =
begin
    input     = hcat(in, state)
    weight    = hcat(layer.wk, layer.wf, layer.wn, layer.wo)
    result    = input * weight
    w_length  = size(layer.wk,2)

    keep      = sigm.(result[:,1:w_length])
    forget    = sigm.(result[:,w_length+1:2*w_length])
    new_state = tanh.(result[:,2*w_length+1:3*w_length])
    out       = tanh.(result[:,3*w_length+1:4*w_length])
    state     = keep .* new_state + forget .* state

state, out
end

zero_state(layer::Recurrent) = zeros(1,size(layer.wk,2))


mutable struct FeedForward
    w::Param
    # b::Param
FeedForward(in_size,layer_size) = new(
    Param(randn(in_size,layer_size)),
    # Param(zeros(1,layer_size)),
)
end

(layer::FeedForward)(in) =
begin
    in * layer.w # + layer.b)
end


mutable struct FeedForward_I
    w::Param
    # b::Param
FeedForward_I(in_size,layer_size) = new(
    Param(Identity(in_size)),
    # Param(zeros(1,layer_size)),
)
end

(layer::FeedForward_I)(in) =
begin
    in * layer.w # + layer.b)
end


prop(model, in) =
begin
    for layer in model
        in = tanh.(layer(in))
    end
in
end

prop(model::Array{FeedForward_I}, in) =
begin
    for layer in model
        in = layer(in)
    end
in
end

prop(model, in; act1=tanh, act2=tanh) =
begin
    for layer in model[1:end-1]
        in = act1.(layer(in))
    end
act2 != nothing ? act2.(model[end](in)) : model[end](in)
end


resize(args...) = reshape(args...)
shape(args...) = size(args...)

expand(arg) = view(arg,[CartesianIndex()],:,:)
stack(args) = vcat(expand.(args)...)
