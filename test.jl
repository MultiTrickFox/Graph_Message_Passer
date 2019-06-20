include("Model.jl")



nn_node1 = [Recurrent(3,3)]
nn_node2 = [Recurrent(3,3)]

lbl_node1 = reshape([0,0,1], 1, 3)
lbl_node2 = reshape([0,1,0], 1, 3)

node1 = Node(nn_node1, lbl_node1)
node2 = Node(nn_node2, lbl_node2)



nn_edge = [FeedForward(3*2,3)]

edge1 = Edge(nn_edge, "some_edge", node1, node2)
edge2 = Edge(nn_edge, "some_edge", node2, node1)


push!(node1.edges, edge1)
push!(node2.edges, edge2)


new_state = update_node_wrt_depths!(node1, depth=1)
println(value(new_state))
