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
    zeros(size(label))
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
    attender::Array{Array{FeedForward}}
    edge_predictor::Array{FeedForward}
    node_predictor::Array{FeedForward}
    node_encodings::Dict
    edge_encodings::Dict

Graph(node_encodings, edge_encodings, hm_attenders) = new(
    [],
    [],
    [[FeedForward(length(node_encodings)+length(edge_encodings), length(node_encodings))] for _ in 1:attender_heads],
    [FeedForward(length(node_encodings)*2, length(edge_encodings))],
    [FeedForward(length(node_encodings), length(node_encodings))],
    node_encodings,
    edge_encodings,
)
end


get_node(graph, node_description) =
    for node in graph.unique_nodes
        if node.description == node_description
            return node
        end
    end

get_edge(graph, edge_description::String) =
    for edge in graph.unique_edges
        if edge.description == edge_description
            return edge
        end
    end

get_edge(graph, node_from::Node, node_to::Node) =
    for edge in graph.unique_edges
        if edge.node_from == node_from && edge.node_to == node_to
            return edge
        end
    end

neighbors_of(node) = [edge.node_to for edge in node.edges]


update_node_wrt_neighbors!(node, attender) =

    if length(node.edges) > 0
        incomings = [prop(edge.nn, hcat(edge.node_to.collected, edge.node_to.label)) for edge in node.edges]
        node.collected = attend(node, attender, incomings)

end

attend(node, attenders, incomings) =
begin

    attendeds = []
    for attender in attenders
        attentions = softmax(vcat([prop(attender, hcat(edge.label, incoming)) for (incoming,edge) in zip(incomings,node.edges)]...), dims=1)
        attended = sum([incoming .* attention for (incoming, attention) in zip(incomings, attentions)])
        push!(attendeds, attended)
    end

sum(attendeds)/length(attenders)
end


update_node_wrt_depths(node, attender) =
begin

    tree = [[node]] # [ [node], [c1, c2], [c1c1, c1c2, c2c1, c2c2] ]
    for _ in 1:propogation_depth-1
        level = []
        for node in tree[end]
            for neighbor in neighbors_of(node)
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
            node.collected = zeros(size(node.label))
        end
    end

root_node_collected
end


predict_edge(graph, node_from::Node, node_to::Node) =
begin

    edge = get_edge(graph, node_from, node_to)
    if edge != nothing
        old_label = edge.label
        edge.label = zeros(size(old_label))
    end
    node_from_collected = update_node_wrt_depths(node_from, graph.attender)
    node_to_collected = update_node_wrt_depths(node_to, graph.attender)
    edge != nothing ? edge.label = old_label : ()

softmax(prop(graph.edge_predictor, hcat(node_from_collected, node_to_collected)))
end


predict_node(graph, node::Node) =
begin

    node_info = node.label
    node.label = zeros(size(node.label))
    node_collected = update_node_wrt_depths(node, graph.attender)
    node.label = node_info

softmax(prop(graph.node_predictor, node_collected))
end
