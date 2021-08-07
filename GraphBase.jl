include("ModelBase.jl")
include("Config.jl")


mutable struct Node
    description::String
    label
    edges # ::Array{Edge}
    collected

Node(description, label, edges=[]) = new(
    description,
    label,
    edges,
    zeros(1, message_size)
)
end


mutable struct Edge
    nn::Array{FeedForward}
    description::String
    label
    node_from::Node
    node_to::Node

Edge(nn, description, label, node_from, node_to) = new(
    nn,
    description,
    label,
    node_from,
    node_to,
)
end


mutable struct Graph
    unique_nodes::Array{Node}
    unique_edges::Array{Edge}
    attender::Array{FeedForward}
    edge_predictor::Array{FeedForward}
    node_predictor::Array{FeedForward}
    node_encodings::Dict
    edge_encodings::Dict

Graph(node_encodings, edge_encodings) = new(
    [],
    [],
    [FeedForward(length(edge_encodings), 1)],
    [FeedForward(message_size*2, length(edge_encodings))],
    [FeedForward(message_size, length(node_encodings))],
    node_encodings,
    edge_encodings,
)
end


get_node(graph, node_description::String) =
    for node in graph.unique_nodes
        if node.description == node_description
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
    for node_from in graph.unique_nodes
        for node_to in graph.unique_nodes
            (edge = get_edge(graph,node_from,node_to)) != nothing ? push!(edges,edge) : ()
        end
    end

edges
end


update_node_wrt_neighbors!(node, attender) =

    if any([edge.label != nothing && edge.node_to.label != nothing for edge in node.edges])
        incomings = vcat([prop(edge.nn, hcat(edge.node_to.collected, edge.node_to.label)) for edge in node.edges if edge.label != nothing && edge.node_to.label != nothing]...)
        attentions = softmax(vcat([prop(attender, edge.label) for edge in node.edges if edge.label != nothing && edge.node_to.label != nothing]...), dims=1)
        node.collected = sum(incomings .* attentions, dims=1)
    end


update_node_wrt_depths(node, attender) =
begin

    tree = [[node]] # [ [node], [c1, c2], [c1c1, c1c2, c2c1, c2c2] ]
    for _ in 1:propogation_depth-1
        level = []
        for node in tree[end]
            for neighbor in [edge.node_to for edge in node.edges if edge.label != nothing && edge.node_to.label != nothing]
                neighbor in level ? () : push!(level, neighbor)
            end
        end
        push!(tree, level)
    end

    for level in reverse(tree)
        for node in level
            update_node_wrt_neighbors!(node, attender)
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
        old_label = edge.label
        edge.label = nothing
    end
    node_from_collected = update_node_wrt_depths(node_from, graph.attender)
    node_to_collected = update_node_wrt_depths(node_to, graph.attender)
    edge != nothing ? edge.label = old_label : ()

softmax(prop(graph.edge_predictor, hcat(node_from_collected, node_to_collected)))
end


predict_node(graph, node::Node) =
begin

    old_label = node.label
    node.label = nothing
    node_collected = update_node_wrt_depths(node, graph.attender)
    node.label = old_label

softmax(prop(graph.node_predictor, node_collected))
end
