include("API.jl")
include("Config.jl")



graph = build_graph("

fox likes dog
fox neutral human

dog likes fox
dog likes human

human likes dog
human likes cat

")



for _ in 1:hm_epochs

    @show predict_edge(graph, "human", "cat")
    # @show embed_node(training_graph, "fox")

    train_for_edge_prediction!(graph,
                               1,
                               learning_rate
                               )

end


println(" ")
