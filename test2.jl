include("Creator.jl")

graph = Graph()

K = [0,0,1]
L = [0,1,0]
Z = [0.2,0.1,0.8]


node_K, node_L = insert!(graph, K, L, "type1")

println(" ")

node_K, node_Z = insert!(graph, K, Z, "type2")

println(" ")

node_L, node_Z = insert!(graph, L, Z, "type2")
