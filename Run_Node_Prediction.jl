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
human neutral fox

human likes dog
human neutral fox

")


question_text = "

X like cat
X like dog

cat like X
dog like X

"






for _ in 1:hm_epochs

    train_for_node_prediction!(graph,
                               1,
                               lr=learning_rate,
                               depth=propogation_depth)

    @show ask_node(graph, question_text, "X", depth=propogation_depth)




end
