include("GraphBase.jl")
include("Config.jl")

using Random: shuffle
using LinearAlgebra: norm


build_graph(graph_string) =
begin

    statements = [split(statement," ") for statement in split(graph_string, "\n") if statement != ""]

    unique_node_names = []
    unique_edge_names = []
    unique_node_types = []

    for (name_node_from, type_node_from, name_edge, name_node_to, type_node_to) in statements

        name_node_from in unique_node_names ? () : push!(unique_node_names, name_node_from)
        name_node_to in unique_node_names ? () : push!(unique_node_names, name_node_to)
        name_edge in unique_edge_names ? () : push!(unique_edge_names, name_edge)
        type_node_from in unique_node_types ? () : push!(unique_node_types, type_node_from)
        type_node_to in unique_node_types ? () : push!(unique_node_types, type_node_to)

    end

    hm_node_names = length(unique_node_names)
    hm_edge_names = length(unique_edge_names)
    hm_node_types = length(unique_node_types)

    graph = Graph()
    graph.node_predictor = [FeedForward(message_size, hm_node_names)]
    graph.edge_predictor = [FeedForward(message_size*2, hm_edge_names)]

    unique_node_names, unique_edge_names, unique_node_types = 1, 1, 1

    for (name_node_from, type_node_from, name_edge, name_node_to, type_node_to) in statements

        if !(name_node_from in keys(graph.node_names))
            graph.node_names[name_node_from] = reshape([i == unique_node_names ? 1.0 : 0.0 for i in 1:hm_node_names], 1, hm_node_names)
            unique_node_names +=1
        end
        if !(name_node_to in keys(graph.node_names))
            graph.node_names[name_node_to] = reshape([i == unique_node_names ? 1.0 : 0.0 for i in 1:hm_node_names], 1, hm_node_names)
            unique_node_names +=1
        end

        if !(type_node_from in keys(graph.node_types))
            graph.node_types[type_node_from] = reshape([i == unique_node_types ? 1.0 : 0.0 for i in 1:hm_node_types], 1, hm_node_types)
            graph.node_nns[type_node_from] = [FeedForward(hm_edge_names, 1)]
            unique_node_types +=1
        end
        if !(type_node_to in keys(graph.node_types))
            graph.node_types[type_node_to] = reshape([i == unique_node_types ? 1.0 : 0.0 for i in 1:hm_node_types], 1, hm_node_types)
            graph.node_nns[type_node_to] = [FeedForward(hm_edge_names, 1)]
            unique_node_types +=1
        end

        if !(name_edge in keys(graph.edge_names))
            graph.edge_names[name_edge] = reshape([i == unique_edge_names ? 1.0 : 0.0 for i in 1:hm_edge_names], 1, hm_edge_names)
            graph.edge_nns[name_edge] = [FeedForward(hm_node_names+message_size, message_size)]
            unique_edge_names +=1
        end

        insert!(graph, (name_node_from, type_node_from, name_edge, name_node_to, type_node_to))

    end

graph
end


insert!(graph, (name_node_from, type_node_from, name_edge, name_node_to, type_node_to); bi_direc=false) =
begin

    node_from_in_graph = false
    for node in graph.nodes
        if node.name == name_node_from
            node_from_in_graph = true
            node_from = node
            break
        end
    end
    if !node_from_in_graph
        node_from = Node(graph.node_nns[type_node_from], name_node_from, type_node_from, graph.node_names[name_node_from])
        push!(graph.nodes, node_from)
    end

    node_to_in_graph = false
    for node in graph.nodes
        if node.name == name_node_to
            node_to_in_graph = true
            node_to = node
            break
        end
    end
    if !node_to_in_graph
        node_to = Node(graph.node_nns[type_node_to], name_node_to, type_node_to, graph.node_names[name_node_to])
        push!(graph.nodes, node_to)
    end

    edge_nn = graph.edge_nns[name_edge]
    edge_encoding = graph.edge_names[name_edge]

    get_edge(graph, node_from, node_to) == nothing ? push!(node_from.edges, Edge(edge_nn, name_edge, edge_encoding, node_from, node_to)) : ()
    bi_direc && get_edge(graph, node_to, node_from) == nothing ? push!(node_to.edges, Edge(edge_nn, name_edge, edge_encoding, node_to, node_from)) : ()

node_from, node_to
end


train_for_edge_prediction!(graph, epochs, lr; edges=all_edges(graph)) =
begin

    losses = []

    for ep in 1:epochs

        loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_node = [zeros(size(getfield(layer, param))) for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.edge_predictor for param in fieldnames(typeof(layer))]

        for edge in edges
            result = @diff sum(cross_entropy(edge.encoding, predict_edge(graph, edge.node_from, edge.node_to)))
            loss += value(result)
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

        println("Epoch $(ep) Loss $(loss)")
        push!(losses, loss)

    end

losses
end

test_for_edge_prediction(graph; edges=all_edges(graph)) =
begin

    count = 0
    for edge in edges
        argmax(predict_edge(graph, edge.node_from, edge.node_to)) == argmax(edge.encoding) ? count+=1 : ()
    end

count/length(edges)
end

predict_edge(graph, node_from::String, node_to::String) =
begin

    node_from = get_node(graph, node_from)
    node_to = get_node(graph, node_to)

    predicted_id = argmax(predict_edge(graph, node_from, node_to))
    for (k,v) in graph.edge_names
        if argmax(v) == predicted_id
            return k
        end
    end

end


train_for_node_prediction!(graph, epochs, lr; nodes=graph.nodes) =
begin

    losses = []

    for ep in 1:epochs

        loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_node = [zeros(size(getfield(layer, param))) for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.node_predictor for param in fieldnames(typeof(layer))]

        for node in nodes
            result = @diff sum(cross_entropy(node.encoding, predict_node(graph, node)))
            loss += value(result)
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

        println("Epoch $(ep) Loss $(loss)")
        push!(losses, loss)

    end

losses
end

test_for_node_prediction(graph; nodes=graph.nodes) =
begin

    count = 0
    for node in nodes
        argmax(predict_node(graph, node)) == argmax(node.encoding) ? count+=1 : ()
    end

count/length(nodes)
end

predict_node(graph, question_graph) =
begin

    question_subject = nothing
    for node_question in question_graph.nodes
        (question_subject = node_question.name) in keys(graph.node_names) ? () : break
    end

    for node_question in question_graph.nodes
        if node_question.name != question_subject
            node_question.encoding = graph.node_names[node_question.name]
            node_question.nn = graph.node_nns[node_question.type]
        end
    end

    for edge_question in all_edges(question_graph)
        edge_question.encoding = graph.edge_names[edge_question.name]
        edge_question.nn = graph.edge_nns[edge_question.name]
    end

    question_node = get_node(question_graph, question_subject)
    question_node.encoding = nothing
    question_node.nn = graph.node_nns[question_node.type]

    question_graph.node_predictor = graph.node_predictor

    predicted_id = argmax(predict_node(question_graph, question_node))
    for (k,v) in graph.node_names
        if argmax(v) == predicted_id
            return k
        end
    end

end


embed_node(graph, node::Node) =
begin

    node_encoding = node.encoding
    node.encoding = nothing
    node_collected = update_node_wrt_depths(node)
    node.encoding = node_encoding

node_collected
end

embed_node(graph, node::String) =
    embed_node(graph, get_node(graph,node))


similarity(embedding1, embedding2; cosine=false) =
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
        println("$(k.node_from.name) <-> $(k.node_to.name) = $(v)")
    end

end


train_for_label_prediction!(graph, epochs, lr, nodes, labels) =
begin

    losses = []

    for ep in 1:epochs

        loss = 0
        grads_edge = [zeros(size(getfield(layer, param))) for nn in values(graph.edge_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_node = [zeros(size(getfield(layer, param))) for nn in values(graph.node_nns) for layer in nn for param in fieldnames(typeof(layer))]
        grads_predictor = [zeros(size(getfield(layer, param))) for layer in graph.edge_predictor for param in fieldnames(typeof(layer))]

        for (node,label) in zip(nodes,labels)
            result = @diff sum(mse(label, prop(graph.label_predictor, embed_node(graph,node))))
            loss += value(result)
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

        println("Epoch $(ep) Loss $(loss)")
        push!(losses, loss)

    end

losses
end

test_for_label_prediction(graph, nodes, labels) =
begin

    distance = 0
    for (node,label) in zip(nodes,labels)
        distance += sum(mse(label, prop(graph.label_predictor, embed_node(graph,node))))
    end

distance
end

predict_label(graph, node::String) =
    prop(graph.label_predictor, embed_node(graph, get_node(graph, node)))
