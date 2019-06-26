include("API.jl")


hm_epochs         = 1_000
learning_rate     = .01
propogation_depth = 2



graph_string = "
A is_afraid B
B is_friend C
C is_neutral D
D is_afraid A
D is_neutral C
C is_afraid B
"


graph = build_graph(graph_string)

train_on!(graph,
          hm_epochs,
          depth=propogation_depth,
          lr=learning_rate)

@show ask(graph, "A", "D")
