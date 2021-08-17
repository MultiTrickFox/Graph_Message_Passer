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



for _ in 1:hm_epochs

    # predict_edge(graph, "human", "dog")
    # embed_node(training_graph, "fox")
    @show test_for_edge_prediction(graph)

    train_for_edge_prediction!(graph, 1, learning_rate)

end

println(" ")
# display_similarities(graph)
# println(" ")
