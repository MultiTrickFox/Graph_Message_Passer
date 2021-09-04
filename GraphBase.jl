include("MathBase.jl")
include("Config.jl")


mutable struct Node
    nn::Array{FeedForward}
    name::String
    type::String
    encoding
    edges # ::Array{Edge}
    collected

Node(nn, name, type, encoding, edges=[]) = new(
    nn,
    name,
    type,
    encoding,
    edges,
    zeros(1, message_size)
)
end


mutable struct Edge
    nn::Array{FeedForward}
    name::String
    encoding
    node_from::Node
    node_to::Node

Edge(nn, name, encoding, node_from, node_to) = new(
    nn,
    name,
    encoding,
    node_from,
    node_to,
)
end


mutable struct Graph
    nodes::Array{Node}
    node_names::Dict
    edge_names::Dict
    node_types::Dict
    node_nns::Dict
    edge_nns::Dict
    node_predictor::Array{FeedForward}
    edge_predictor::Array{FeedForward}
    label_predictor::Array{FeedForward}


Graph(;node_names=Dict(), edge_names=Dict(), node_types=Dict(), node_nns=Dict(), edge_nns=Dict()) = new(
    [],
    node_names,
    edge_names,
    node_types,
    node_nns,
    edge_nns,
    [],
    [],
    [],
)
end


get_node(graph, node_name::String) =
    for node in graph.nodes
        if node.name == node_name
            return node
        end
    end

get_edge(graph, node_from::Node, node_to::Node) =
    for edge in node_from.edges
        if edge.node_to == node_to
            return edge
        end
    end

neighbors_of(node) = [edge.node_to for edge in node.edges]

all_edges(graph) =
begin

    edges = []

    for node_from in graph.nodes
        for node_to in graph.nodes
            node_from != node_to && (edge = get_edge(graph,node_from,node_to)) != nothing ? push!(edges,edge) : ()
        end
    end

    for node in graph.nodes
        (edge = get_edge(graph,node,node)) != nothing ? push!(edges,edge) : ()
    end

edges
end


update_node_wrt_neighbors!(node) =
begin

    edges = [edge for edge in node.edges if edge.encoding != nothing && edge.node_to.encoding != nothing]

    if length(edges) > 0

        unknown_encoding = (node.encoding == nothing)
        if unknown_encoding
            for edge in node.edges
                if (encoding = edge.node_to.encoding) != nothing
                    node.encoding = zeros(size(encoding))
                    break
                end
            end
        end

        incomings = vcat([prop(edge.nn, hcat(edge.node_to.collected,edge.node_to.encoding)) for edge in edges]...)
        attentions = softmax(vcat([prop(node.nn, hcat(node.encoding,edge.encoding); act2=nothing) for edge in edges]...))
        node.collected = sum(incomings .* attentions, dims=1)

        unknown_encoding ? node.encoding = nothing : ()

    end

end


update_node_wrt_depths(node) =
begin

    tree = [[node]]
    for _ in 1:propogation_depth-1
        level = []
        for node in tree[end]
            for neighbor in [edge.node_to for edge in node.edges if edge.encoding != nothing && edge.node_to.encoding != nothing]
                neighbor in level ? () : push!(level, neighbor)
            end
        end
        push!(tree, level)
    end

    for level in reverse(tree)
        for node in level
            update_node_wrt_neighbors!(node)
        end
    end
    root_node_collected = node.collected

    for level in tree
        for node in level
            node.collected = zeros(1, message_size)
        end
    end

root_node_collected
end


predict_edge(graph, node_from::Node, node_to::Node) =
begin

    edge = get_edge(graph, node_from, node_to)
    if edge != nothing
        old_encoding = edge.encoding
        edge.encoding = nothing
    end
    node_from_collected = update_node_wrt_depths(node_from)
    node_to_collected = update_node_wrt_depths(node_to)
    edge != nothing ? edge.encoding = old_encoding : ()

softmax(prop(graph.edge_predictor, hcat(node_from_collected, node_to_collected); act2=nothing))
end


predict_node(graph, node::Node) =
begin

    old_encoding = node.encoding
    node.encoding = nothing
    node_collected = update_node_wrt_depths(node)
    node.encoding = old_encoding

softmax(prop(graph.node_predictor, node_collected; act2=nothing))
end
