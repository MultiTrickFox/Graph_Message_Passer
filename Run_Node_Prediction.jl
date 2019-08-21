include("API.jl")


hm_epochs         = 1_000
learning_rate     = 0.2
propogation_depth = 2


training_graph = build_graph("

fox likes dog
fox neutral human
fox dislikes cat

dog likes fox
dog likes human
dog dislikes cat

human likes dog
human neutral fox
human likes cat

cat likes human
cat dislikes dog
cat dislikes fox

")


question_graph = build_graph("

fox likes X
X likes human

")


question_subject = "X"



for _ in 1:hm_epochs

    train_for_node_prediction!(training_graph,
                               1,
                               lr=learning_rate,
                               depth=propogation_depth)


    @show predict_node(training_graph, question_graph, question_subject, depth=propogation_depth)


end
