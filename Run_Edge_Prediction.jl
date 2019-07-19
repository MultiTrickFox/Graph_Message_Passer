include("API.jl")


hm_epochs         = 1_000
learning_rate     = .5
propogation_depth = 2


# graph = build_graph("
#
# fox neutral human
# fox dislikes cat
# fox likes dog
#
# human likes cat
# human likes dog
#
# cat dislikes fox
# cat dislikes dog
# cat likes human
#
# dog likes human
# dog likes fox
# dog dislikes cat
#
# ")

graph = build_graph("

fox likes dog
fox neutral human

dog likes fox
dog likes human

human likes dog
human neutral fox

human likes dog
human neutral fox

")

for _ in 1:hm_epochs

    train_for_edge_prediction!(graph,
                               1,
                               depth=propogation_depth,
                               lr=learning_rate)

    @show predict_edge(graph, "human", "fox", depth=propogation_depth)


end
