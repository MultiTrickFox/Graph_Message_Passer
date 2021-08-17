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
human mammal neutral fox animal
human mammal likes cat animal

cat animal likes human mammal
cat animal dislikes dog animal
cat animal dislikes fox animal

")


question_graph = build_graph("

fox animal likes X animal
X animal likes human mammal
human mammal likes X animal
X animal dislikes cat animal

human mammal neutral fox animal
human mammal likes cat animal

cat animal dislikes X animal

")



for _ in 1:hm_epochs

    # predict_node(graph, question_graph)
    # embed_node(graph, "fox")
    @show test_for_node_prediction(graph)

    train_for_node_prediction!(graph, 1, learning_rate)

end

println(" ")
# display_similarities(graph)
# println(" ")
