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



@show test_for_edge_prediction(graph)

for i in 1:hm_epochs

    println("train: $(train_for_edge_prediction!(graph, learning_rate))")

    i%test_per_epoch == 0 ? println("\ttest: $(test_for_edge_prediction(graph))") : ()

end

println(" ")



# predict_edge(graph, "human", "dog")
# embed_node(graph, "fox")
# display_similarities(graph)
