include("API.jl")


hm_epochs         = 1_000
learning_rate     = .5
propogation_depth = 2


graph = build_graph("

fox likes dog
fox neutral human

dog likes fox
dog likes human

human likes dog

")



for _ in 1:hm_epochs

    train_for_edge_prediction!(graph,
                               1,
                               depth=propogation_depth,
                               lr=learning_rate)


    @show predict_edge(graph, "human", "fox", depth=propogation_depth)


end
