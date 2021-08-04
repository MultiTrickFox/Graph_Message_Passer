include("GraphBase.jl")
include("Config.jl")

using Random: shuffle
using LinearAlgebra: norm


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

    graph = Graph(node_encodings, edge_encodings)

    for statement in split(graph_string, "\n")
        if statement != ""
            description_node_from, description_edge, description_node_to = split(statement, " ")
            label_node_from, label_edge, label_node_to = node_encodings[description_node_from], edge_encodings[description_edge], node_encodings[description_node_to]
            insert!(graph, (description_node_from, label_node_from), (description_edge, label_edge), (description_node_to, label_node_to))
        end
    end

graph
end


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
            node_from_in_graph = true
            node_from = node
            break
        end
    end
    if !node_from_in_graph
        node_from = Node(description_node_from, label_node_from)
        push!(graph.unique_nodes, node_from)
    end

    node_to_in_graph = false
    for node in graph.unique_nodes
        if node.label == label_node_to
            node_to_in_graph = true
            node_to = node
            break
        end
    end
    if !node_to_in_graph
        node_to = Node(description_node_to, label_node_to)
        push!(graph.unique_nodes, node_to)
    end

    edge_in_graph = false
    for edge in graph.unique_edges
        if edge.label == label_edge
            edge_in_graph = true
            edge_nn = edge.nn
            break
        end
    end
    edge_in_graph ? () : edge_nn = [FeedForward(label_size_node+message_size, message_size)]

    if bi_direc

        edge1 = Edge(edge_nn, description_edge, label_edge, node_from, node_to)
        edge2 = Edge(edge_nn, description_edge, label_edge, node_to, node_from)
        edge_in_graph ? () : push!(graph.unique_edges, edge1)
        push!(node_from.edges, edge1)
        push!(node_to.edges, edge2)

    else

        edge = Edge(edge_nn, description_edge, label_edge, node_from, node_to)
        edge_in_graph ? () : push!(graph.unique_edges, edge)
        push!(node_from.edges, edge)

    end

node_from, node_to
end


train_for_edge_prediction!(graph, epochs, lr) =

    for ep in 1:epochs

        ep_loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for edge in graph.unique_edges for layer in edge.nn for param in fieldnames(typeof(layer))]
        grads_attender = [zeros(size(getfield(layer, param))) for layer in graph.attender for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.edge_predictor for param in fieldnames(typeof(layer))]

        for node_from in graph.unique_nodes
            for node_to in graph.unique_nodes
                if (edge = get_edge(graph,node_from,node_to)) != nothing
                    result = @diff sum(cross_entropy(edge.label, predict_edge(graph, node_from, node_to)))
                    ep_loss += value(result)
                    grads_edge += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for edge in graph.unique_edges for layer in edge.nn for param in fieldnames(typeof(layer))]
                    grads_attender += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for layer in graph.attender for param in fieldnames(typeof(layer))]
                    grads_predictor += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for layer in graph.edge_predictor for param in fieldnames(typeof(layer))]
                end
            end
        end

        for edge in graph.unique_edges
            for layer in edge.nn
                for (param,grad) in zip(fieldnames(typeof(layer)),grads_edge)
                    setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
                end
            end
        end
        for layer in graph.attender
            for (param,grad) in zip(fieldnames(typeof(layer)),grads_attender)
                setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
            end
        end
        for layer in graph.edge_predictor
            for (param,grad) in zip(fieldnames(typeof(layer)),grads_predictor)
                setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
            end
        end

        println("Epoch $(ep) Loss $(ep_loss)")

    end

test_for_edge_prediction(graph) =
begin

    count = 0
    count2 = 0
    for node_from in graph.unique_nodes
        for node_to in graph.unique_nodes
            if (edge = get_edge(graph,node_from,node_to)) != nothing
                count2 +=1
                argmax(predict_edge(graph, node_from, node_to)) == argmax(edge.label) ? count+=1 : ()
            end
        end
    end

count/count2
end

predict_edge(graph, node_from::String, node_to::String) =
begin

    node_from = get_node(graph, node_from)
    node_to = get_node(graph, node_to)

    predicted_id = argmax(predict_edge(graph, node_from, node_to))
    for edge in graph.unique_edges
        if argmax(edge.label) == predicted_id
            return edge.description
        end
    end

end


train_for_node_prediction!(graph, epochs, lr) =

    for ep in 1:epochs

        ep_loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for edge in graph.unique_edges for layer in edge.nn for param in fieldnames(typeof(layer))]
        grads_attender = [zeros(size(getfield(layer, param))) for layer in graph.attender for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.node_predictor for param in fieldnames(typeof(layer))]

        for node in shuffle(graph.unique_nodes)
            result = @diff sum(cross_entropy(node.label, predict_node(graph, node)))
            ep_loss += value(result)
            grads_edge += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for edge in graph.unique_edges for layer in edge.nn for param in fieldnames(typeof(layer))]
            grads_attender += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for layer in graph.attender for param in fieldnames(typeof(layer))]
            grads_predictor += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for layer in graph.node_predictor for param in fieldnames(typeof(layer))]
        end

        for edge in graph.unique_edges
            for layer in edge.nn
                for (param,grad) in zip(fieldnames(typeof(layer)),grads_edge)
                    setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
                end
            end
        end
        for layer in graph.attender
            for (param,grad) in zip(fieldnames(typeof(layer)),grads_attender)
                setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
            end
        end
        for layer in graph.node_predictor
            for (param,grad) in zip(fieldnames(typeof(layer)),grads_predictor)
                setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
            end
        end

        println("Epoch $(ep) Loss $(ep_loss)")

    end

test_for_node_prediction(graph) =
begin

    count = 0
    for node in graph.unique_nodes
        argmax(predict_node(graph, node)) == argmax(node.label) ? count+=1 : ()
    end

count/length(graph.unique_nodes)
end

predict_node(graph, question_graph) =
begin

    question_subject = nothing
    for node_question in question_graph.unique_nodes
        node_found = false
        for node_original in graph.unique_nodes
            if node_question.description == node_original.description
                node_found = true
                break
            end
        end
        if !node_found
            question_subject = node_question.description
            break
        end
    end

    for node_question in question_graph.unique_nodes
        for node_original in graph.unique_nodes
            if node_question.description == node_original.description
                node_question.label = node_original.label
                break
            end
        end
    end

    for edge_question in vcat([node.edges for node in question_graph.unique_nodes]...)
        for edge_original in graph.unique_edges
            if edge_question.description == edge_original.description
                edge_question.nn = edge_original.nn
                edge_question.label = edge_original.label
                break
            end
        end
    end

    question_node = get_node(question_graph, question_subject)
    question_node.label = zeros(size(graph.unique_nodes[1].label))

    question_node_collected = update_node_wrt_depths(question_node, graph.attender)

    predicted_id = argmax(prop(graph.node_predictor, question_node_collected))
    for node in graph.unique_nodes
        if argmax(node.label) == predicted_id
            return node.description
        end
    end

end


embed_node(graph, node::Node) =
begin

    node_label = node.label
    node.label = zeros(size(node_label))
    node_collected = update_node_wrt_depths(node, graph.attender)
    node.label = node_label

node_collected
end

embed_node(graph, node::String) =
    embed_node(get_node(graph,node))


similarity(embedding1, embedding2; cosine=true) =
    cosine ? sum(embedding1.*embedding2)/(norm(embedding1)*norm(embedding2)) :
        sqrt(sum((embedding1.-embedding2).^2))

similarity(graph, node1, node2) =
    similarity(embed_node(graph,node1), embed_node(graph,node2))


display_similarities(graph) =
begin

    scores = Dict()

    for node_from in graph.unique_nodes
        for node_to in graph.unique_nodes
            already_calculated = false
            for (k,v) in scores
                if k.node_from == node_to && k.node_to == node_from
                    already_calculated = true
                    break
                end
            end
            if !already_calculated && (edge = get_edge(graph,node_from,node_to)) != nothing
                scores[edge] = similarity(graph, node_from, node_to)
            end
        end
    end

    for (k,v) in sort(scores; byvalue=true)
        println("$(k.node_from.description) <-> $(k.node_to.description) = $(v)")
    end

end
