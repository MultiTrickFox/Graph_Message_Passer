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


labels = Dict()
labels[get_node(graph,"fox")] = tanh.(randn(1,2))
labels[get_node(graph,"dog")] = tanh.(randn(1,2))
labels[get_node(graph,"human")] = tanh.(randn(1,2))
labels[get_node(graph,"cat")] = tanh.(randn(1,2))

graph.label_predictor = [FeedForward(message_size, size(collect(values(labels))[end])[end])]



@show test_for_label_prediction(graph, keys(labels), values(labels))

for i in 1:hm_epochs

    println("train: $(train_for_label_prediction!(graph, learning_rate, keys(labels), values(labels)))")

    i%test_per_epoch == 0 ? println("\ttest: $(test_for_label_prediction(graph, keys(labels), values(labels)))") : ()

end

println(" ")



# predict_label(graph, "human")
# embed_node(graph, "fox")
# display_similarities(graph)



# binary_cross_entropy(label, prop(predictor, update_node_wrt_depths(node), act2=sigm))
# cross_entropy(label, softmax(prop(predictor, update_node_wrt_depths(node), act2=nothing)))
# mse(label, prop(predictor, update_node_wrt_depths(node)))
