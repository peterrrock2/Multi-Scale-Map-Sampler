import Pkg
push!(LOAD_PATH, "..");

using RandomNumbers
using MultiScaleMapSampler

pctGraphPath = joinpath("..", "test", "test_graphs", "NC_pct21.json")
precinct_name = "prec_id"
county_name = "county"
population_col = "pop2020cen"
num_dists = 14
rng_seed = 454190
pop_dev = 0.02
gamma = 0 #0 is uniform on forests; 1 is uniform on partitions
steps = 10
edge_weights= "connections"
output_file_path = "./atlas_gamma"*string(gamma)*".jsonl"

nodeData = Set([precinct_name, county_name, population_col])

base_graph = BaseGraph(
    pctGraphPath, 
    population_col, 
    inc_node_data=nodeData,
    edge_weights=edge_weights
);
graph = MultiLevelGraph(base_graph, [county_name, precinct_name]);

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, 0.01))
add_constraint!(constraints, ConstrainDiscontinuousTraversals(graph))
add_constraint!(constraints, MaxCoarseNodeSplits(num_dists+1))

rng = PCG.PCGStateOneseq(UInt64, rng_seed)
partition = MultiLevelPartition(graph, constraints, num_dists; rng=rng);

proposal = build_forest_recom2(constraints)
measure = Measure(gamma)
# to add elements to the measure
# push_measure!(measure, get_isoperimetric_score, 0.45)

# output_file_path = "./atlas_gamma"*string(gamma)*".jsonl"
# output_file_io = smartOpen(output_file_path, "w")
# writer = Writer(measure, constraints, partition, output_file_io)
writer = Writer(measure, constraints, partition, stdout)


run_metropolis_hastings!(
    partition, 
    proposal, 
    measure, 
    steps, 
    rng,
    writer=writer, 
    output_freq=1
);
