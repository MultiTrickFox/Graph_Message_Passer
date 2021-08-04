include("API.jl")
include("Config.jl")



graph = build_graph("

fox likes dog
fox neutral human
fox dislikes cat

dog likes fox
dog likes human
dog dislikes cat

human likes dog
human likes cat
human likes fox

cat neutral human
cat dislikes dog
cat dislikes fox

")



for _ in 1:hm_epochs

    # @show predict_edge(graph, "human", "dog")
    # @show embed_node(training_graph, "fox")

    @show test_for_edge_prediction(graph)

    train_for_edge_prediction!(graph,
                               1,
                               learning_rate
                               )

end


println(" ")
