include("Models.jl")


lbl_size = 3


mutable struct Graph
    unique_nodes::Array{Node}
    unique_edges::Array{Edge}

Graph() = new(
    [],
    [],
)
end


insert!(graph, label_node_from, label_node_to, description_edge; bidirec_edge=false) =
begin

    node_from_in_graph = false

    for node in graph.unique_nodes
        if node.label == label_node_from
            node_from_in_graph = true

            println("node_from found in graph.")

            node_from = node
            break
        end
    end

    if !node_from_in_graph

        println("node_from not found in graph.")

        node_from = Node([Recurrent(lbl_size, lbl_size)], label_node_from)
        push!(graph.unique_nodes, node_from)

        println("added $label_node_from to graph.")

    end

    node_to_in_graph = false

    for node in graph.unique_nodes
        if node.label == label_node_to
            node_to_in_graph = true

            println("node_to found in graph.")

            node_to = node
            break
        end
    end

    if !node_to_in_graph

        println("node_to not found in graph.")

        node_to = Node([Recurrent(lbl_size, lbl_size)], label_node_to)
        push!(graph.unique_nodes, node_to)

        println("added $label_node_to to graph.")

    end

    edge_in_graph = false

    for edge in graph.unique_edges
        if edge.description == description_edge
            edge_in_graph = true

            println("edge found in graph")

            edge_nn = edge.nn
            break
        end
    end

    if !edge_in_graph
        edge_nn = [FeedForward(lbl_size, lbl_size)]
    end

    if bidirec_edge

        edge1 = Edge(edge_nn, description_edge, node_from, node_to)
        edge2 = Edge(edge_nn, description_edge, node_to, node_from)

        if !edge_in_graph

            println("edge not found in graph")

            push!(graph.unique_edges, edge1)
        end

        push!(node_from.edges, edge1)
        push!(node_to.edges, edge2)

    else

        edge = Edge(edge_nn, description_edge, node_from, node_to)

        if !edge_in_graph

            println("edge not found in graph")

            push!(graph.unique_edges, edge)
        end

        push!(node_from.edges, edge)

    end

node_from, node_to
end
