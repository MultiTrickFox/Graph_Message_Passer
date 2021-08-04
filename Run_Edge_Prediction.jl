include("API.jl")


hm_epochs         = 100
learning_rate     = .5
propogation_depth = 2


graph = build_graph("

fox likes dog
fox neutral human

dog likes fox
dog likes human

human likes dog
human likes cat

")



for _ in 1:hm_epochs

    @show predict_edge(graph, "human", "cat", depth=propogation_depth)

    train_for_edge_prediction!(graph,
                               1,
                               depth=propogation_depth,
                               lr=learning_rate)

end


println(" ")
