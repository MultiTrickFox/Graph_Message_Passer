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


train_for_edge_prediction!(graph, epochs; depth=1, lr=.001) =
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


predict_edge(graph, node_from::String, node_to::String; depth=1) =
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


train_for_node_prediction!(graph, epochs; depth=1, lr=.001) =
    for ep in 1:epochs

        ep_loss = 0
        grads = [zeros(size(getfield(layer, param))) for node in graph.unique_nodes for layer in node.nn for param in fieldnames(typeof(layer))]

        # get grads

        for node in shuffle(graph.unique_nodes)

            result = @diff sum(cross_entropy(node.label, predict_node(graph, node, depth=depth)))

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


predict_node(graph, question_graph, question_subject; depth=1) =
begin

    for node_question in question_graph.unique_nodes
        for node_original in graph.unique_nodes
            if node_question.description == node_original.description
                node_question.nn = node_original.nn
                node_question.label = node_original.label
                node_question.collected = zeros(size(node_question.label))
            end
        end
    end

    for edge_question in question_graph.unique_edges
        for edge_original in graph.unique_edges
            if edge_question.description == edge_original.description
                edge_question.nn = edge_original.nn
                edge_question.label = edge_original.label
            end
        end
    end

    question_graph.attender = graph.attender

    size_lbl = length(graph.unique_nodes[1].label)
    question_node = get_node(question_graph, question_subject)
    question_node.nn = [FeedForward_I(size_lbl, size_lbl)]
    question_node.label = zeros(1, size_lbl)
    question_node.collected = zeros(1, size_lbl)

    for node in question_graph.unique_nodes
        for edge in node.edges
            for edge2 in question_graph.unique_edges
                if edge.description == edge2.description
                    edge.nn = edge2.nn
                    break
                end
            end
        end
    end


    ###

    _, attended = update_node_wrt_depths!(question_node, question_graph.attender, depth=depth)

    picked_id = argmax(attended)
    picked_node = nothing
    for node in graph.unique_nodes
        if argmax(node.label) == picked_id
            picked_node = node
            break
        end
    end


picked_node.description
end
