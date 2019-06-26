using AutoGrad: Param, @diff, value, grad


sigm(x) = 1 / (1 + exp(-x))
tanh(x) = 2 * sigm(2*x) - 1

softmax(list) = (exp_list -> exp_list./sum(exp_list))(exp.(list))
cross_entropy(label, prediction) = -(label .* log.(prediction))
mse(label, prediction) = (label - prediction) .^2


mutable struct Recurrent
    state::Param
    wf1::Param
    wf2::Param
    bf::Param
    wk1::Param
    wk2::Param
    bk::Param
    wi::Param
    bi::Param
Recurrent(in_size,layer_size) = new(
    Param(zeros(1,layer_size)),
    Param(randn(in_size,layer_size)),
    Param(randn(layer_size,layer_size)),
    Param(zeros(1,layer_size)),
    Param(randn(in_size,layer_size)),
    Param(randn(layer_size,layer_size)),
    Param(zeros(1,layer_size)),
    Param(randn(in_size,layer_size)),
    Param(zeros(1,layer_size)),
)
end

(layer::Recurrent)(in) =
begin
    focus  = sigm.(in * layer.wf1 + layer.state * layer.wf2 + layer.bf)
    keep   = sigm.(in * layer.wk1 + layer.state * layer.wk2 + layer.bk)
    interm = tanh.(in * layer.wi  + layer.state .* focus    + layer.bi)

layer.state = Param(keep .* interm + (1 .- keep) .* layer.state)
end

reset_state!(layer::Recurrent) =
begin
    layer.state = Param(zeros(size(layer.state)))
end


mutable struct FeedForward
    w::Param
    b::Param
FeedForward(in_size,layer_size) = new(
    Param(randn(in_size,layer_size)),
    Param(zeros(1,layer_size)),
)
end

(layer::FeedForward)(in) =
begin
    in * layer.w + layer.b
end


prop(model, in) =
begin
    for layer in model
        in = tanh.(layer(in))
    end
    
in
end
