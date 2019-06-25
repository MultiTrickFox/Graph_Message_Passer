include("Models.jl")

# "
#
# Example Graph:
#
#     K [0 0 1] -> edge_type_1 [0 1] -> L [0 1 0]
#
#     L [0 1 0] -> edge_type_2 [1 0] -> Z [1 0 0]
#
#     Z [1 0 0] <-> edge_type_1 [0 1] <-> K [0 0 1]
#
# "
#
#
# graph = Graph()
#
#
#
# K_description = "K"
# K_label = [0 0 1]
#
# L_description = "L"
# L_label = [0 1 0]
#
# Z_description = "Z"
# Z_label = [0.2 0.1 0.8]
#
#
# Edge1_description = "Edge1"
# Edge1_label = [0 1]
#
# Edge2_description = "Edge2"
# Edge2_label = [1 0]
#
#
#
# insert!(graph,
#         (K_description, K_label),
#         (L_description, L_label),
#         (Edge1_description, Edge1_label),
#         ) ; println(" ")
#
# insert!(graph,
#         (L_description, L_label),
#         (Z_description, Z_label),
#         (Edge2_description, Edge2_label),
#         ) ; println(" ")
#
# insert!(graph,
#         (Z_description, Z_label),
#         (K_description, K_label),
#         (Edge1_description, Edge1_label),
#         bi_direc=true
#         ) ; println(" ")
#
#
#
# node_K = get_node(graph,"K")
# node_L = get_node(graph, "L")
#
#
# collected = value(update_node_wrt_depths!(node_K, depth=2))
#
# println(collected)
#
#
# predicted_edge_K_to_L = predict_edge(graph, node_K, node_L)
#
# println(value(predicted_edge_K_to_L))
#
#
# train_on!(graph, 10, depth=2, lr=.01)
#
# edge_between_L_K = predicted_edge(graph, predict_edge(graph, node_L, node_K))
#
# @show edge_between_L_K



# println("new tests")
# println("new tests")
# println(" ")


node_encodings = Dict()
edge_encodings = Dict()


node_encodings["A"] = [0 0 1 0]
node_encodings["B"] = [0 1 0 0]
node_encodings["C"] = [1 0 0 0]
node_encodings["D"] = [0 0 0 1]

edge_encodings["is_afraid"] = [0 1 0]
edge_encodings["is_friend"] = [1 0 0]
edge_encodings["is_neutral"] = [0 0 1]



graph_string = "
A is_afraid B
B is_friend C
C is_neutral D
D is_afraid A
D is_neutral C
C is_afraid B
"



graph = build_graph(graph_string, node_encodings, edge_encodings)

train_on!(graph, 20, depth=2)
