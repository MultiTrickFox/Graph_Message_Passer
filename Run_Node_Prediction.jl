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



for _ in 1:hm_epochs

    # @show predict_node(graph, question_graph)
    # @show embed_node(graph, "fox")
    @show test_for_node_prediction(graph)

    train_for_node_prediction!(graph, 1, learning_rate)

end

println(" ")
# display_similarities(graph)
# println(" ")
