include("ModelBase.jl")


mutable struct Node
    nn::Array{FeedForward}
    description::String
    label
    edges # ::Array{Edge} # cmon julia u can do circular declaration. u should.
    collected

Node(nn, description, label, edges=[]) = new(
    nn,
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
    predictor::Array{FeedForward}
    node_encodings::Dict
    edge_encodings::Dict

Graph(node_encodings, edge_encodings, hm_attenders) = new(
    [],
    [],
    [[FeedForward(length(node_encodings)*2, length(node_encodings))] for _ in 1:hm_attenders],
    [FeedForward(length(node_encodings)*2, length(edge_encodings))],
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


insert!(graph,
        (description_node_from, label_node_from),
        (description_edge, label_edge),
        (description_node_to, label_node_to),
        bi_direc=false) =

begin

    label_size_node = length(graph.node_encodings)
    label_size_edge = length(graph.edge_encodings)

    node_from_in_graph = false
    for node in graph.unique_nodes
        if node.label == label_node_from
            # debug ? println("$(node.description) found in graph.") : ()
            node_from_in_graph = true
            node_from = node
            break
        end
    end
    if !node_from_in_graph
        # debug ? println("$(description_node_from) not in graph.") : ()
        node_from = Node([FeedForward(label_size_node, label_size_node)], description_node_from, label_node_from)
        push!(graph.unique_nodes, node_from)
    end

    node_to_in_graph = false
    for node in graph.unique_nodes
        if node.label == label_node_to
            # debug ? println("$(node.description) found in graph.") : ()
            node_to_in_graph = true
            node_to = node
            break
        end
    end
    if !node_to_in_graph
        # debug ? println("$(description_node_to) not in graph.") : ()
        node_to = Node([FeedForward(label_size_node, label_size_node)], description_node_to, label_node_to)
        push!(graph.unique_nodes, node_to)
    end

    edge_in_graph = false
    for edge in graph.unique_edges
        if edge.label == label_edge
            # debug ? println("$(edge.description) found in graph.") : ()
            edge_in_graph = true
            edge_nn = edge.nn
            break
        end
    end
    if !edge_in_graph
        # debug ? println("$(description_edge) not in graph.") : ()
        edge_nn = [FeedForward(label_size_node*2, label_size_node)]
    end

    if bi_direc

        edge1 = Edge(edge_nn, description_edge, label_edge, node_from, node_to)
        edge2 = Edge(edge_nn, description_edge, label_edge, node_to, node_from)

        if !edge_in_graph
            push!(graph.unique_edges, edge1)
        end

        push!(node_from.edges, edge1)
        push!(node_to.edges, edge2)

    else

        edge = Edge(edge_nn, description_edge, label_edge, node_from, node_to)

        if !edge_in_graph
            push!(graph.unique_edges, edge)
        end

        push!(node_from.edges, edge)

    end

node_from, node_to
end


update_node_wrt_neighbors!(node, attender) =
begin

    incomings = [prop(edge.nn, hcat(edge.node_to.collected, edge.node_to.label)) for edge in node.edges]
    attended = pass_from_heads(node, attender, incomings)
    node.collected = prop(node.nn, attended)

attended
end


pass_from_heads(node, attenders, incomings) =
begin

    attendeds = []
    for attender in attenders
        attentions_pre = [prop(attender, hcat(node.label, incoming)) for incoming in incomings]
        attentions = softmax(vcat(attentions_pre...), dims=1)
        attended = sum([incoming .* attention for (incoming, attention) in zip(incomings, attentions)])
        push!(attendeds, attended)
    end

sum(attendeds)/length(attenders)
end


update_node_wrt_depths!(node, attender; depth=1) = # TODO : bang or not, look at the whole
begin

    tree = [[node]] # [ [node], [c1, c2], [c1c1, c1c2, c2c1, c2c2] ]

    for _ in 1:depth-1

        level = []

        for node in tree[end]

            push!(level, neighbors_of(node))

        end

        push!(tree, vcat(level...))

    end

    root_node_attended = 0

    for level in reverse(tree)

        for node in level

            root_node_attended = update_node_wrt_neighbors!(node, attender)

        end

    end

    root_node_collected = node.collected

    for level in tree

        for node in level

            node.collected = zeros(size(node.label))

        end

    end

root_node_collected, root_node_attended
end



# FOLLOWING ARE CALLED BY OUTER MODULES #


predict_edge(graph, node_from, node_to; depth=1) =
begin
    edge = get_edge(graph, node_from, node_to)
    if edge != nothing
        old_label = edge.label
        edge.label = zeros(size(old_label))
    end
    encoding_node_from, _ = update_node_wrt_depths!(node_from, graph.attender, depth=depth)
    encoding_node_to, _ = update_node_wrt_depths!(node_to, graph.attender, depth=depth)
    edge != nothing ? edge.label = old_label : ()

softmax(prop(graph.predictor, hcat(encoding_node_from, encoding_node_to)))
end


predict_node(graph, node; depth=1) =
begin

    node_info = node.label
    node.label = zeros(size(node.label))
    _, attended = update_node_wrt_depths!(node, graph.attender, depth=depth)
    node.label = node_info

softmax(attended)
end
