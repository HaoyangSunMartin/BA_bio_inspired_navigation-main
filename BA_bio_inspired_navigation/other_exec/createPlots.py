from plotting.plotThesis import *
from system.bio_model.placecellModel import PlaceCellNetwork
from system.bio_model.cognitivemapModel import CognitiveMapNetwork
import operator


def plt_grid_cell_decoder():
    model = "spike_detection/"  # "spike_detection/", "linear_lookahead/", "phase_offset_detector/"

    errors = np.load("experiments/" + model + "error_array" + ".npy")
    positions = np.load("experiments/" + model + "position_array" + ".npy")
    vectors = np.load("experiments/" + model + "vectors_array" + ".npy")

    evaluate = 8000  # pick between 3501 and 8000, 6300 for 2nd look-ahead
    errors_evaluate = np.linalg.norm(positions[:, evaluate] + vectors[:, evaluate], axis=1)
    positions_evaluate = positions[:, evaluate]
    vectors_evaluate = vectors[:, evaluate]

    filter_threshold = 0.5
    delete = np.where(errors_evaluate < filter_threshold, False, True)
    errors_evaluate = np.delete(errors_evaluate, delete)
    positions_evaluate = np.delete(positions_evaluate, delete, axis=0)
    vectors_evaluate = np.delete(vectors_evaluate, delete, axis=0)

    filter_ratio = len(errors_evaluate) / 50
    print(model, "| Evaluate: ", str(evaluate), " | Threshold: ", str(filter_threshold), " | Filter ration: ", filter_ratio)

    plot_positions(positions_evaluate, vectors_evaluate)
    plot_vector_navigation_error(errors_evaluate)
    plot_vector_angle_error(positions_evaluate, vectors_evaluate)
    plot_vector_distance_error(positions_evaluate, vectors_evaluate)

    run = 0
    plot_error_single_run(positions[run], vectors[run])
    plot_angles_single_run(positions[run], vectors[run])

    factor_0 = 1  # 0.27 for phase_offset, 1 for spike_detection
    factors = np.linspace(1e-7, 2, num=100)
    mse = {}
    for factor in factors:
        true_distance = np.linalg.norm(positions[:, 3501:], axis=2)
        computed_distance = (np.linalg.norm(vectors[:, 3501:], axis=2) / factor_0) * factor
        mean_squared_error = np.square(true_distance - computed_distance).mean(axis=None)
        mse[factor] = mean_squared_error

    factor_best = min(mse.items(), key=operator.itemgetter(1))[0]
    print(factor_best)

    plot_mean_squared_error(mse)

    # factor_best = 0.28528535675675676
    run = 16
    plot_distances_single_run(positions[run], vectors[run] * factor_best / factor_0)


def plot_cognitive_map():
    from_data = True
    dt = 1e-2

    # initialize Place Cell Model
    pc_network = PlaceCellNetwork(from_data=from_data)
    # initialize Cognitive Map Model
    cognitive_map = CognitiveMapNetwork(dt, from_data=from_data)
    cognitive_map_plot(pc_network, cognitive_map)


def plot_trajectory_maze():
    door = "door_5"
    errors = np.load("experiments/maze_navigation/" + door + "/data/" + "error_array" + ".npy")
    positions = np.load("experiments/maze_navigation/" + door + "/data/" + "position_array" + ".npy")
    vectors = np.load("experiments/maze_navigation/" + door + "/data/" + "vectors_array" + ".npy")

    run = 0
    xy_coordinates = positions[run]
    plot_trajectory(xy_coordinates, door)


plot_trajectory_maze()

