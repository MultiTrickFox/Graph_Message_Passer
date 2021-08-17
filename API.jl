include("GraphBase.jl")
include("Config.jl")

using Random: shuffle
using LinearAlgebra: norm


build_graph(graph_string) =
begin

    statements = [split(statement," ") for statement in split(graph_string, "\n") if statement != ""]

    unique_node_descriptions = []
    unique_edge_descriptions = []
    unique_node_types = []

    for (description_node_from, type_node_from, description_edge, description_node_to, type_node_to) in statements

        description_node_from in unique_node_descriptions ? () : push!(unique_node_descriptions, description_node_from)
        description_node_to in unique_node_descriptions ? () : push!(unique_node_descriptions, description_node_to)
        description_edge in unique_edge_descriptions ? () : push!(unique_edge_descriptions, description_edge)
        type_node_from in unique_node_types ? () : push!(unique_node_types, type_node_from)
        type_node_to in unique_node_types ? () : push!(unique_node_types, type_node_to)

    end

    hm_node_descriptions = length(unique_node_descriptions)
    hm_edge_descriptions = length(unique_edge_descriptions)
    hm_node_types = length(unique_node_types)

    graph = Graph()
    graph.node_predictor = [FeedForward(message_size, hm_node_descriptions)]
    graph.edge_predictor = [FeedForward(message_size*2, hm_edge_descriptions)]

    unique_node_ctr, unique_edge_ctr, unique_node_ctr2 = 1, 1, 1

    for (description_node_from, type_node_from, description_edge, description_node_to, type_node_to) in statements

        if !(description_node_from in keys(graph.node_encodings))
            graph.node_encodings[description_node_from] = reshape([i == unique_node_ctr ? 1.0 : 0.0 for i in 1:hm_node_descriptions], 1, hm_node_descriptions)
            unique_node_ctr +=1
        end
        if !(type_node_from in keys(graph.node_encodings2))
            graph.node_encodings2[type_node_from] = reshape([i == unique_node_ctr2 ? 1.0 : 0.0 for i in 1:hm_node_types], 1, hm_node_types)
            graph.node_nns[type_node_from] = [FeedForward(hm_edge_descriptions, 1)]
            unique_node_ctr2 +=1
        end

        if !(description_node_to in keys(graph.node_encodings))
            graph.node_encodings[description_node_to] = reshape([i == unique_node_ctr ? 1.0 : 0.0 for i in 1:hm_node_descriptions], 1, hm_node_descriptions)
            unique_node_ctr +=1
        end
        if !(type_node_to in keys(graph.node_encodings2))
            graph.node_encodings2[type_node_to] = reshape([i == unique_node_ctr2 ? 1.0 : 0.0 for i in 1:hm_node_types], 1, hm_node_types)
            graph.node_nns[type_node_to] = [FeedForward(hm_edge_descriptions, 1)]
            unique_node_ctr2 +=1
        end

        if !(description_edge in keys(graph.edge_encodings))
            graph.edge_encodings[description_edge] = reshape([i == unique_edge_ctr ? 1.0 : 0.0 for i in 1:hm_edge_descriptions], 1, hm_edge_descriptions)
            graph.edge_nns[description_edge] = [FeedForward(hm_node_descriptions+message_size, message_size)]
            unique_edge_ctr +=1
        end

        insert!(graph, (description_node_from, type_node_from, description_edge, description_node_to, type_node_to))

    end

graph
end


insert!(graph, (description_node_from, type_node_from, description_edge, description_node_to, type_node_to); bi_direc=false) =
begin

    node_from_in_graph = false
    for node in graph.nodes
        if node.description == description_node_from
            node_from_in_graph = true
            node_from = node
            break
        end
    end
    if !node_from_in_graph
        node_from = Node(graph.node_nns[type_node_from], description_node_from, type_node_from, graph.node_encodings[description_node_from])
        push!(graph.nodes, node_from)
    end

    node_to_in_graph = false
    for node in graph.nodes
        if node.description == description_node_to
            node_to_in_graph = true
            node_to = node
            break
        end
    end
    if !node_to_in_graph
        node_to = Node(graph.node_nns[type_node_to], description_node_to, type_node_to, graph.node_encodings[description_node_to])
        push!(graph.nodes, node_to)
    end

    edge_nn = graph.edge_nns[description_edge]
    edge_label = graph.edge_encodings[description_edge]

    if bi_direc

        edge1 = Edge(edge_nn, description_edge, edge_label, node_from, node_to)
        edge2 = Edge(edge_nn, description_edge, edge_label, node_to, node_from)
        push!(node_from.edges, edge1)
        push!(node_to.edges, edge2)

    else

        edge = Edge(edge_nn, description_edge, edge_label, node_from, node_to)
        push!(node_from.edges, edge)

    end

node_from, node_to
end


train_for_edge_prediction!(graph, epochs, lr; edges=all_edges(graph)) =

    for ep in 1:epochs

        ep_loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_node = [zeros(size(getfield(layer, param))) for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.edge_predictor for param in fieldnames(typeof(layer))]

        for edge in edges
            result = @diff sum(cross_entropy(edge.label, predict_edge(graph, edge.node_from, edge.node_to)))
            ep_loss += value(result)
            grads_edge += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
            grads_node += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
            grads_predictor += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for layer in graph.edge_predictor for param in fieldnames(typeof(layer))]
        end

        for nn in values(graph.edge_nns)
            for layer in nn
                for (param,grad) in zip(fieldnames(typeof(layer)),grads_edge)
                    setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
                end
            end
        end
        for nn in values(graph.node_nns)
            for layer in nn
                for (param,grad) in zip(fieldnames(typeof(layer)),grads_node)
                    setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
                end
            end
        end
        for layer in graph.edge_predictor
            for (param,grad) in zip(fieldnames(typeof(layer)),grads_predictor)
                setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
            end
        end

        println("Epoch $(ep) Loss $(ep_loss)")

    end

test_for_edge_prediction(graph; edges=all_edges(graph)) =
begin

    count = 0
    for edge in edges
        argmax(predict_edge(graph, edge.node_from, edge.node_to)) == argmax(edge.label) ? count+=1 : ()
    end

count/length(edges)
end

predict_edge(graph, node_from::String, node_to::String) =
begin

    node_from = get_node(graph, node_from)
    node_to = get_node(graph, node_to)

    predicted_id = argmax(predict_edge(graph, node_from, node_to))
    for (k,v) in graph.edge_encodings
        if argmax(v) == predicted_id
            return k
        end
    end

end


train_for_node_prediction!(graph, epochs, lr; nodes=graph.nodes) =

    for ep in 1:epochs

        ep_loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_node = [zeros(size(getfield(layer, param))) for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.node_predictor for param in fieldnames(typeof(layer))]

        for node in nodes
            result = @diff sum(cross_entropy(node.label, predict_node(graph, node)))
            ep_loss += value(result)
            grads_edge += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
            grads_node += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
            grads_predictor += [(g = grad(result, getfield(layer, param))) == nothing ? zeros(size(getfield(layer, param))) : g for layer in graph.node_predictor for param in fieldnames(typeof(layer))]
        end

        for nn in values(graph.edge_nns)
            for layer in nn
                for (param,grad) in zip(fieldnames(typeof(layer)),grads_edge)
                    setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
                end
            end
        end
        for nn in values(graph.node_nns)
            for layer in nn
                for (param,grad) in zip(fieldnames(typeof(layer)),grads_node)
                    setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
                end
            end
        end
        for layer in graph.node_predictor
            for (param,grad) in zip(fieldnames(typeof(layer)),grads_predictor)
                setfield!(layer, param, Param(getfield(layer, param) -lr*grad))
            end
        end

        println("Epoch $(ep) Loss $(ep_loss)")

    end

test_for_node_prediction(graph; nodes=graph.nodes) =
begin

    count = 0
    for node in nodes
        argmax(predict_node(graph, node)) == argmax(node.label) ? count+=1 : ()
    end

count/length(nodes)
end

predict_node(graph, question_graph) =
begin

    question_subject = nothing
    for node_question in question_graph.nodes
        node_found = false
        for node_original in graph.nodes
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

    for node_question in question_graph.nodes
        if node_question.description != question_subject
            node_question.label = graph.node_encodings[node_question.description]
            node_question.nn = graph.node_nns[node_question.type]
        end
    end

    for edge_question in all_edges(question_graph)
        edge_question.label = graph.edge_encodings[edge_question.description]
        edge_question.nn = graph.edge_nns[edge_question.description]
    end

    question_node = get_node(question_graph, question_subject)
    question_node.label = nothing
    question_node.nn = graph.node_nns[question_node.type]

    question_node_collected = update_node_wrt_depths(question_node)

    predicted_id = argmax(prop(graph.node_predictor, question_node_collected))
    for node in graph.nodes
        if argmax(node.label) == predicted_id
            return node.description
        end
    end

end


embed_node(graph, node::Node) =
begin

    node_label = node.label
    node.label = nothing
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

    for node_from in graph.nodes
        for node_to in graph.nodes
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

    for (k,v) in sort(collect(scores); by=x->x[2])
        println("$(k.node_from.description) <-> $(k.node_to.description) = $(v)")
    end

end
