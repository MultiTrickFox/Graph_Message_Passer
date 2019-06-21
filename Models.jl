using AutoGrad: Param, @diff, value, grad


sigm(x) = 1 / (1 + exp(-x))
tanh(x) = 2 * sigm(2*x) - 1


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
        in = layer(in)
    end
in
end



mutable struct Node
    nn::Array{Recurrent}
    label
    edges

Node(nn, label, edges=[]) = new(
    nn,
    label,
    edges,
)
end


mutable struct Edge
    nn::Array{FeedForward}
    description::String
    node_from::Node
    node_to::Node

Edge(nn, description, node_from, node_to) = new(
    nn,
    description,
    node_from,
    node_to,
)
end


reset_state!(node) =
    for layer in node.nn
        layer.state = Param(reshape(node.label, 1, length(node.label)))
    end


neighbors_of(node) = [edge.node_to for edge in node.edges]


update_node_wrt_neighbors!(node) =
begin

        incoming = [prop(edge.nn, edge.node_to.nn[end].state)
                        for edge in node.edges]
        prop(node.nn, sum(incoming))

        # node.nn[end].state = Param(normalize(node.nn[end].state))

node.nn[end].state
end


update_node_wrt_depths!(node; depth=1) =
begin

    tree = [[node]] # [ [node], [c1, c2], [c1c1, c1c2, c2c1, c2c2] ]

    for _ in 1:depth-1

        level = []

        for node in tree[end]

            push!(level, neighbors_of(node))

        end

        push!(tree, vcat(level...))

    end

    for level in reverse(tree)

        for node in level

            update_node_wrt_neighbors!(node)

        end

    end

node.nn[end].state
end


normalize(array) = array / sqrt(sum(array .^2))
