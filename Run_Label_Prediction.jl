include("API.jl")
include("Config.jl")



graph = build_graph("

fox animal likes dog animal
fox animal neutral human mammal
fox animal dislikes cat animal

dog animal likes fox animal
dog animal likes human mammal
dog animal dislikes cat animal

human mammal likes dog animal
human mammal likes cat animal
human mammal likes fox animal

cat animal neutral human mammal
cat animal dislikes dog animal
cat animal dislikes fox animal

")


labels = Dict()
labels[get_node(graph,"fox")] = randn(1,2)
labels[get_node(graph,"dog")] = randn(1,2)
labels[get_node(graph,"human")] = randn(1,2)
labels[get_node(graph,"cat")] = randn(1,2)

graph.label_predictor = [FeedForward(message_size, size(collect(values(labels))[end])[end])]



for _ in 1:hm_epochs

    # predict_label(graph, "human")
    # embed_node(graph, "fox")
    @show test_for_label_prediction(graph, keys(labels), values(labels))

    train_for_label_prediction!(graph, 1, learning_rate, keys(labels), values(labels))

end

println(" ")
# display_similarities(graph)
# println(" ")
