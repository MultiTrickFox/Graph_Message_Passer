include("GraphBase.jl")

using Random: shuffle



# @ HYPERPARAMS

hm_attenders = 4

# END



build_graph(graph_string) =
begin

    node_encodings = Dict()
    edge_encodings = Dict()
    unique_node_ctr, unique_edge_ctr = 1, 1

    for statement in split(graph_string, "\n")
        if statement != ""
            description_node_from, description_edge, description_node_to = split(statement, " ")

            for description_node in (description_node_from, description_node_to)
                if !(description_node in keys(node_encodings))
                    node_encodings[description_node] = "placeholder"
                    unique_node_ctr +=1
                end
            end

            if !(description_edge in keys(edge_encodings))
                edge_encodings[description_edge] = "placeholder"
                unique_edge_ctr +=1
            end

        end
    end

    label_size_node = length(node_encodings)
    label_size_edge = length(edge_encodings)

    node_encodings = Dict()
    edge_encodings = Dict()
    unique_node_ctr, unique_edge_ctr = 1, 1

    for statement in split(graph_string, "\n")
        if statement != ""
            description_node_from, description_edge, description_node_to = split(statement, " ")

            for description_node in (description_node_from, description_node_to)
                if !(description_node in keys(node_encodings))
                    encoding_node = reshape([i == unique_node_ctr ? 1.0 : 0.0 for i in 1:label_size_node], 1, label_size_node)
                    node_encodings[description_node] = encoding_node
                    unique_node_ctr +=1
                end
            end

            if !(description_edge in keys(edge_encodings))
                encoding_edge = reshape([i == unique_edge_ctr ? 1.0 : 0.0 for i in 1:label_size_edge], 1, label_size_edge)
                edge_encodings[description_edge] = encoding_edge
                unique_edge_ctr +=1
            end

        end
    end

    graph = Graph(node_encodings, edge_encodings, hm_attenders)

    for statement in split(graph_string, "\n")
        if statement != ""
            description_node_from, description_edge, description_node_to = split(statement, " ")
            label_node_from, label_edge, label_node_to = node_encodings[description_node_from], edge_encodings[description_edge], node_encodings[description_node_to]
            insert!(graph, (description_node_from, label_node_from), (description_edge, label_edge), (description_node_to, label_node_to))
        end
    end

graph
end


train_on!(graph, epochs; depth=1, lr=.001) =
    for ep in 1:epochs

        ep_loss = 0
        grads = [zeros(size(getfield(layer, param))) for node in graph.unique_nodes for layer in node.nn for param in fieldnames(typeof(layer))]

        # get grads

        for edge in shuffle(graph.unique_edges)

            result = @diff sum(cross_entropy(edge.label, predict_edge(graph, edge.node_from, edge.node_to, depth=depth)))

            ep_loss += value(result)

            grads += [grad(result, getfield(layer, param)) == nothing ? zeros(size(getfield(layer, param))) : grad(result, getfield(layer, param)) for node in graph.unique_nodes for layer in node.nn for param in fieldnames(typeof(layer))]

        end

        # update params

        ctr = 0
        for node in graph.unique_nodes
            for layer in node.nn
                for param in fieldnames(typeof(layer))
                    ctr +=1

                    setfield!(layer, param, Param(getfield(layer, param) - lr * grads[ctr]))

                end
            end
        end

        # display

        println("Epoch $(ep) Loss $(ep_loss)")

    end


ask(graph, node_from, node_to; depth=1) =
begin
    node_from = get_node(graph, node_from)
    node_to = get_node(graph, node_to)
    predicted_id = argmax(predict_edge(graph, node_from, node_to, depth=depth))

    for edge in graph.unique_edges
        if argmax(edge.label) == predicted_id
            return edge.description
        end
    end
end
