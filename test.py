from flask import Flask, request, jsonify
import numpy as np
import os
import pulp
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/api/find_path', methods=['POST'])
def find_path():
    try:
        data = request.get_json()
        ka = int(data.get("start_point"))
        kt = int(data.get("end_point"))
        mode = data.get("mode")

        if ka is None or kt is None:
            return jsonify({"error": "Missing start or end point"}), 400

        # File paths
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        distances_file = os.path.join(BASE_PATH, 'Penteli_Distances.csv')
        xyz_file_path = os.path.join(BASE_PATH, 'Penteli_CentroidNodes_coordinates.xyz')
        min_pavement_width_side1 = os.path.join(BASE_PATH, 'diagonal_table_Side1_minpavementwidth.csv')
        min_pavement_width_side2 = os.path.join(BASE_PATH, 'diagonal_table_Side2_minpavementwidth.csv')
        horizontal_slope_side1 = os.path.join(BASE_PATH, 'diagonal_table_Side1_horizontalslope.csv')
        horizontal_slope_side2 = os.path.join(BASE_PATH, 'diagonal_table_Side2_horizontalslope.csv')
        kerb_side1 = os.path.join(BASE_PATH, 'diagonal_table_Side1_kerb.csv')
        kerb_side2 = os.path.join(BASE_PATH, 'diagonal_table_Side2_kerb.csv')
        max_kerb_slope_side1 = os.path.join(BASE_PATH, 'diagonal_table_Side1_maxkerbslope.csv')
        max_kerb_slope_side2 = os.path.join(BASE_PATH, 'diagonal_table_Side2_maxkerbslope.csv')
        tactile_paving_side1 = os.path.join(BASE_PATH, 'diagonal_table_Side1_tactilepaving.csv')
        tactile_paving_side2 = os.path.join(BASE_PATH, 'diagonal_table_Side2_tactilepaving.csv')
        # Read the accessibility standards CSV file
        AccessibilityStandards_df = pd.read_csv("/Users/kkase/Desktop/AccessibilityStandards.csv",
                                                dtype={"Min_Pavement_Width": float, "Max_Horizontal_Slope": float,
                                                       "Max_Kerb_Slope": float})
        AccessibilityStandards_df.set_index("Parameter", inplace=True)  # Set the index
        AccessibilityStandards_df.index = AccessibilityStandards_df.index.str.strip()  # Remove extra spaces

        # Set accessibility file paths based on the mode
        if mode == 'pedestrian':
            accessibility_side1_file = os.path.join(BASE_PATH, 'diagonal_table_Side1_Pedestrians_v2.csv')
            accessibility_side2_file = os.path.join(BASE_PATH, 'diagonal_table_Side2_Pedestrians_v2.csv')
        elif mode == 'mobilityImpaired':
            accessibility_side1_file = os.path.join(BASE_PATH, 'diagonal_table_Side1_MobilityImpaired_v2.csv')
            accessibility_side2_file = os.path.join(BASE_PATH, 'diagonal_table_Side2_MobilityImpaired_v2.csv')
        elif mode == 'visuallyImpaired':
            accessibility_side1_file = os.path.join(BASE_PATH, 'diagonal_table_Side1_VisuallyImpaired_v2.csv')
            accessibility_side2_file = os.path.join(BASE_PATH, 'diagonal_table_Side2_VisuallyImpaired_v2.csv')
        else:
            return jsonify({"error": "Invalid mode specified"}), 400

        # Load Data
        D = np.loadtxt(distances_file, delimiter=',', skiprows=1)
        A_classification_side1 = np.loadtxt(accessibility_side1_file, delimiter=',', skiprows=1)
        A_classification_side2 = np.loadtxt(accessibility_side2_file, delimiter=',', skiprows=1)
        # Load additional data files
        Min_pavement_width_side1 = np.loadtxt(min_pavement_width_side1, delimiter=',', skiprows=1)
        Min_pavement_width_side2 = np.loadtxt(min_pavement_width_side2, delimiter=',', skiprows=1)
        Horizontal_slope_side1 = np.loadtxt(horizontal_slope_side1, delimiter=',', skiprows=1)
        Horizontal_slope_side2 = np.loadtxt(horizontal_slope_side2, delimiter=',', skiprows=1)
        Kerb_side1 = np.loadtxt(kerb_side1, delimiter=',', skiprows=1, dtype=str)
        Kerb_side2 = np.loadtxt(kerb_side2, delimiter=',', skiprows=1, dtype=str)
        Max_kerb_slope_side1 = np.loadtxt(max_kerb_slope_side1, delimiter=',', skiprows=1)
        Max_kerb_slope_side2 = np.loadtxt(max_kerb_slope_side2, delimiter=',', skiprows=1)
        Tactile_paving_side1 = np.loadtxt(tactile_paving_side1, delimiter=',', skiprows=1, dtype=str)
        Tactile_paving_side2 = np.loadtxt(tactile_paving_side2, delimiter=',', skiprows=1, dtype=str)

        # Extract the values from the DataFrame
        Min_Pavement_Width = AccessibilityStandards_df.loc["Min_Pavement_Width", "Value"]
        Max_Horizontal_Slope = AccessibilityStandards_df.loc["Max_Horizontal_Slope", "Value"]
        Max_Kerb_Slope = AccessibilityStandards_df.loc["Max_Kerb_Slope", "Value"]
        Kerb_Requirement = AccessibilityStandards_df.loc["Kerb_Requirement", "Value"].strip().lower()
        Tactile_Paving_Requirement = str(
        AccessibilityStandards_df.loc["Tactile_Paving_Requirement", "Visually_Impaired"]).strip().lower()

        # Handle invalid values in accessibility matrices
        A_classification_side1 = np.nan_to_num(A_classification_side1, nan=0, posinf=0, neginf=0)
        A_classification_side2 = np.nan_to_num(A_classification_side2, nan=0, posinf=0, neginf=0)

        # Check for NaN or invalid values
        if np.isnan(A_classification_side1).any() or np.isnan(A_classification_side2).any():
            return jsonify({"error": "Accessibility matrices contain NaN values"}), 400
        if np.isinf(A_classification_side1).any() or np.isinf(A_classification_side2).any():
            return jsonify({"error": "Accessibility matrices contain infinite values"}), 400

        # Combine accessibility matrices
        combined_accessibility = np.maximum(A_classification_side1, A_classification_side2)

        # Ensure shapes match
        if D.shape != A_classification_side1.shape or D.shape != A_classification_side2.shape:
            raise ValueError("Distances matrix and Accessibility matrix dimensions must match.")

        # Check if there are any accessible paths
        if np.sum(combined_accessibility) == 0:
            return jsonify({"error": "No accessible paths found for the given mode"}), 400

        # Accessible distances only
        accessible_D = np.where(combined_accessibility > 0, D, 0)

        # Read node positions from .xyz file
        positions = {}
        with open(xyz_file_path, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split()
                if len(parts) == 4:
                    node, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    positions[node] = (x, y)

        # Swapping x and y for visualization purposes
        swapped_positions = {node: (y, x) for node, (x, y) in positions.items()}

        # Sets
        V = {i: i for i in range(1, len(D) + 1)}  # Set of all nodes
        P = {i: i for i in V if i != ka and i != kt}  # Set of all nodes apart from the initial and the final one

        # Edges for the fully accessible path
        A = {(ka, i) for i in V if accessible_D[ka - 1, i - 1] > 0}.union(
            {(i, j) for i in P for j in P if i != j and accessible_D[i - 1, j - 1] > 0}).union(
            {(i, kt) for i in V if accessible_D[i - 1, kt - 1] > 0})

        # Edges for alternative path (includes all paths)
        A_alt = {(ka, i) for i in V if D[ka - 1, i - 1] > 0}.union(
            {(i, j) for i in P for j in P if i != j and D[i - 1, j - 1] > 0}).union(
            {(i, kt) for i in V if D[i - 1, kt - 1] > 0})

        # First Optimization: Find the shortest fully accessible path
        model = pulp.LpProblem("Shortest_Path", pulp.LpMinimize)

        # Variables
        x = pulp.LpVariable.dicts("X", (V, V), cat="Binary")  # x[i, j] is 1 if edge (i, j) is used
        c = pulp.LpVariable.dicts("c", (V, V), lowBound=0)  # Cost of edge (i, j)
        C = pulp.LpVariable("C", lowBound=0)  # Total cost

        # Objective
        model += pulp.lpSum(c[i][j] for (i, j) in A)

        # Constraints
        model += pulp.lpSum(x[ka][j] for j in V if (ka, j) in A) == 1  # Outbound flow from ka
        model += pulp.lpSum(x[j][kt] for j in V if (j, kt) in A) == 1  # Inbound flow to kt
        for j in P:
            model += pulp.lpSum(x[i][j] for i in V if (i, j) in A) == pulp.lpSum(x[j][i] for i in V if (j, i) in A)  # Flow balance
        for (i, j) in A:
            model += c[i][j] == accessible_D[i - 1][j - 1] * x[i][j]  # Cost definition
        model += pulp.lpSum(c[i][j] for (i, j) in A) == C  # Total cost

        # Solve the model
        try:
            model.solve()
        except pulp.PulpSolverError as e:
            logger.error(f"Error solving model: {str(e)}")
            # If model fails, skip to model3
            model_status = "Infeasible"
        else:
            model_status = pulp.LpStatus[model.status]

        # Initialize variables for model results
        C_accessible = float('inf')
        shortest_path_edges = []

        if model_status == "Optimal":
            C_accessible = pulp.value(C)
            shortest_path_edges = [(i, j) for (i, j) in A if pulp.value(x[i][j]) > 0.5]
        else:
            logger.info("Model1 is infeasible or failed. Proceeding to model3.")

        # Second Optimization: Allow one inaccessible edge but ensure it's shorter than the first solution
        model2 = pulp.LpProblem("Alternative_Path", pulp.LpMinimize)

        # Variables
        x2 = pulp.LpVariable.dicts("X2", (V, V), cat="Binary")
        c2 = pulp.LpVariable.dicts("c2", (V, V), lowBound=0)
        C2 = pulp.LpVariable("C2", lowBound=0)
        y = pulp.LpVariable.dicts("Y", (V, V), cat="Binary")

        # Objective
        model2 += pulp.lpSum(c2[i][j] for (i, j) in A_alt)

        # Constraints
        model2 += pulp.lpSum(x2[ka][j] for j in V if (ka, j) in A_alt) == 1
        model2 += pulp.lpSum(x2[j][kt] for j in V if (j, kt) in A_alt) == 1
        for j in P:
            model2 += pulp.lpSum(x2[i][j] for i in V if (i, j) in A_alt) == pulp.lpSum(x2[j][i] for i in V if (j, i) in A_alt)
        model2 += pulp.lpSum(y[i][j] for (i, j) in A_alt if combined_accessibility[i - 1][j - 1] == 0) == 1
        for (i, j) in A_alt:
            model2 += y[i][j] == x2[i][j] * (1 - combined_accessibility[i - 1][j - 1])
            model2 += c2[i][j] == D[i - 1][j - 1] * x2[i][j]
        model2 += pulp.lpSum(c2[i][j] for (i, j) in A_alt) == C2
        model2 += C2 <= C_accessible

        # Solve the model
        try:
            model2.solve()
        except pulp.PulpSolverError as e:
            logger.error(f"Error solving model2: {str(e)}")
            # If model2 fails, skip to model3
            model2_status = "Infeasible"
        else:
            model2_status = pulp.LpStatus[model2.status]

        # Initialize variables for model2 results
        alternative_path_edges = []
        inaccessible_edges = []

        if model2_status == "Optimal":
            alternative_path_edges = [(i, j) for (i, j) in A_alt if pulp.value(x2[i][j]) > 0.5]
            inaccessible_edges = [(i, j) for (i, j) in alternative_path_edges if combined_accessibility[i - 1][j - 1] == 0]
        else:
            logger.info("Model2 is infeasible or failed. Proceeding to model3.")

        # Third Optimization: Allow an inaccessible path when the first and second models are infeasible
        model3 = pulp.LpProblem("Second_Alternative_Path", pulp.LpMinimize)

        # Variables
        x3 = pulp.LpVariable.dicts("X3", (V, V), cat="Binary")
        c3 = pulp.LpVariable.dicts("c3", (V, V), lowBound=0)
        C3 = pulp.LpVariable("C3", lowBound=0)

        # Objective
        model3 += pulp.lpSum(c3[i][j] for (i, j) in A_alt)

        # Constraints
        model3 += pulp.lpSum(x3[ka][j] for j in V if (ka, j) in A_alt) == 1
        model3 += pulp.lpSum(x3[j][kt] for j in V if (j, kt) in A_alt) == 1
        for j in P:
            model3 += pulp.lpSum(x3[i][j] for i in V if (i, j) in A_alt) == pulp.lpSum(x3[j][i] for i in V if (j, i) in A_alt)
        for (i, j) in A_alt:
            model3 += c3[i][j] == D[i - 1][j - 1] * x3[i][j]
        model3 += pulp.lpSum(c3[i][j] for (i, j) in A_alt) == C3

        # Solve the model
        try:
            model3.solve()
        except pulp.PulpSolverError as e:
            logger.error(f"Error solving model3: {str(e)}")
            return jsonify({"error": f"Solver error: {str(e)}"}), 500

        second_alternative_path_edges = []
        if pulp.LpStatus[model3.status] == "Optimal":
            second_alternative_path_edges = [(i, j) for (i, j) in A_alt if pulp.value(x3[i][j]) > 0.5]
        else:
            logger.error("Model3 is also infeasible or failed. No valid paths found.")
            return jsonify({"error": "No valid paths found for the given constraints"}), 400

        # Identify inaccessible edges for model3
        inaccessible_edges_model3 = [(i, j) for (i, j) in second_alternative_path_edges if
                                     combined_accessibility[i - 1][j - 1] == 0]

        # Function to reconstruct ordered path
        def reconstruct_path(start, edges):
            path = [start]
            edge_dict = {i: j for i, j in edges}
            while path[-1] in edge_dict:
                path.append(edge_dict[path[-1]])
            return path

        # Reconstruct paths only if the model is feasible
        ordered_shortest_path = []
        if model_status == "Optimal":
            ordered_shortest_path = reconstruct_path(ka, shortest_path_edges)

        ordered_alternative_path = []
        if model2_status == "Optimal":
            ordered_alternative_path = reconstruct_path(ka, alternative_path_edges)

        ordered_second_alternative_path = []
        if pulp.LpStatus[model3.status] == "Optimal":
            ordered_second_alternative_path = reconstruct_path(ka, second_alternative_path_edges)

        speed_mps = 1.38889  # 5 km/h
        total_distance = pulp.value(C) if model_status == "Optimal" else float('inf')
        total_time = total_distance / speed_mps if total_distance != float('inf') else float('inf')
        total_distance_alternative = pulp.value(C2) if model2_status == "Optimal" else float('inf')
        total_time_alternative = total_distance_alternative / speed_mps if total_distance_alternative != float('inf') else float('inf')
        total_distance_second_alternative = pulp.value(C3) if pulp.LpStatus[model3.status] == "Optimal" else float('inf')
        total_time_second_alternative = total_distance_second_alternative / speed_mps if total_distance_second_alternative != float('inf') else float('inf')

        # Function to check obstacles for inaccessible edges
        def check_obstacles(edges, mode):
            obstacles = {}
            for (i, j) in edges:
                reasons = []

                # Check min_pavement_width for all modes
                if mode == 'mobilityImpaired':
                    if Min_pavement_width_side1[i - 1, j - 1] < Min_Pavement_Width and Min_pavement_width_side2[
                        i - 1, j - 1] < Min_Pavement_Width:
                        reasons.append("narrowPavement")
                elif mode in ['visuallyImpaired', 'pedestrian']:
                    if Min_pavement_width_side1[i - 1, j - 1] <= 0 and Min_pavement_width_side2[i - 1, j - 1] <= 0:
                        reasons.append("NoPavement")

                # Check horizontal_slope, kerb, and max_kerb_slope only for mobilityImpaired
                if mode == 'mobilityImpaired':
                    if Horizontal_slope_side1[i - 1, j - 1] >= Max_Horizontal_Slope and Horizontal_slope_side2[i - 1, j - 1] >= Max_Horizontal_Slope:
                        reasons.append("SteepHorizontalSlope")
                    if Kerb_side1[i - 1, j - 1].strip().lower() != Kerb_Requirement and Kerb_side2[
                        i - 1, j - 1].strip().lower() != Kerb_Requirement:
                        reasons.append("NoDroppedKerb")
                    if Max_kerb_slope_side1[i - 1, j - 1] >= Max_Kerb_Slope and Max_kerb_slope_side2[
                        i - 1, j - 1] >= Max_Kerb_Slope:
                        reasons.append("HighKerbSlope")

                # Check tactile paving only for visuallyImpaired
                if mode == 'visuallyImpaired':
                    if Tactile_paving_side1[i - 1, j - 1].strip().lower() != Tactile_Paving_Requirement and \
                            Tactile_paving_side2[i - 1, j - 1].strip().lower() != Tactile_Paving_Requirement:
                        reasons.append("NoTactilePaving")

                # Use a string key instead of a tuple
                obstacles[f"{i}-{j}"] = reasons  # Store obstacles for this edge
            return obstacles

        # Check obstacles for inaccessible edges
        obstacles = check_obstacles(inaccessible_edges, mode)

        # Check obstacles for inaccessible edges in model3
        obstacles_model3 = check_obstacles(inaccessible_edges_model3, mode)

        def replace_infinity(obj):
            """Recursively replace infinity values with a large number or null."""
            if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
                return None
            elif isinstance(obj, list):
                return [replace_infinity(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: replace_infinity(value) for key, value in obj.items()}
            return obj

        # Prepare the response
        response = {
            "start_node": ka,
            "end_node": kt,
            "total_distance": round(total_distance, 2) if total_distance != float('inf') else None,
            "total_time": round(total_time, 2) if total_time != float('inf') else None,
            "path": [(ordered_shortest_path[i], ordered_shortest_path[i + 1]) for i in range(len(ordered_shortest_path) - 1)] if ordered_shortest_path else [],
            "alternative_path": [(ordered_alternative_path[i], ordered_alternative_path[i + 1]) for i in range(len(ordered_alternative_path) - 1)] if ordered_alternative_path else [],
            "inaccessible_edges": inaccessible_edges,
            "obstacles": obstacles,
            "second_alternative_path": [(ordered_second_alternative_path[i], ordered_second_alternative_path[i + 1]) for i in range(len(ordered_second_alternative_path) - 1)] if ordered_second_alternative_path else [],
            "inaccessible_edges_model3": inaccessible_edges_model3,
            "obstacles_model3": obstacles_model3,
            "total_distance_alternative": round(total_distance_alternative, 2) if total_distance_alternative != float('inf') else None,
            "total_time_alternative": round(total_time_alternative, 2) if total_time_alternative != float('inf') else None,
            "total_distance_second_alternative": round(total_distance_second_alternative, 2) if total_distance_second_alternative != float('inf') else None,
            "total_time_second_alternative": round(total_time_second_alternative, 2) if total_time_second_alternative != float('inf') else None
        }
        response = replace_infinity(response)
        logger.debug(f"Response: {response}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)