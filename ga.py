import bpy, bmesh
from deap import base, creator, tools, algorithms
import random
import numpy as np
import time
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from collections import defaultdict, Counter
start_time = time.time()


# Basic Utils

def calculate_distance(vertex1, vertex2):
    return np.linalg.norm(np.array(vertex1) - np.array(vertex2))


def get_sorted_models_from_collection(collection):
    if collection:
        return sorted([obj for obj in collection.objects if obj.type == 'MESH'], key=lambda x: x.name)
    else:
        return []


def get_sorted_models_from_collection_group(collection_group):
    collection_models = defaultdict(list)
    for collection in collection_group.children: 
        sorted_models = get_sorted_models_from_collection(collection)
        collection_name = collection.name.split('.')[0]  # OVERAY-12.xx에서OVERAY-12 추출
        collection_models[collection_name].extend(sorted_models)
    
    return collection_models


def get_vertex_positions(obj_name):
    obj = bpy.data.objects[obj_name]
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return [v.to_tuple() for v in vertices]


def get_transformed_vertex_positions(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    transformed_vertices = [v.co for v in mesh.vertices]
    obj_eval.to_mesh_clear()

    return [v.to_tuple() for v in transformed_vertices]


def calculate_total_distance2(transformed_vertices, reference_vertices, selected_indices):
    transformed_array = np.array([transformed_vertices[i] for i in selected_indices])
    reference_array = np.array([reference_vertices[i] for i in selected_indices])
    distances = np.linalg.norm(transformed_array - reference_array, axis=1)
    return np.sum(distances)


# Lattice Utils

def clear_lattice_modifier():
    for obj in bpy.data.objects:
        for modifier in obj.modifiers:
            if modifier.type == 'LATTICE':
                obj.modifiers.remove(modifier)
    bpy.context.view_layer.update()


def create_and_setup_lattice(name, location, scale, subdivisions):
    bpy.ops.object.add(type='LATTICE', location=location)
    lattice_obj = bpy.context.object
    lattice_obj.name = name
    lattice_obj.scale = scale
    
    lattice_data = lattice_obj.data
    lattice_data.points_u = subdivisions
    lattice_data.points_v = subdivisions
    lattice_data.points_w = subdivisions
    
    return lattice_obj


def reset_lattice(lattice_obj):  # deform the lattice to the original shape
    for point in lattice_obj.data.points:
        point.co_deform.x = point.co.x
        point.co_deform.y = point.co.y
        point.co_deform.z = point.co.z


def apply_lattice_to_object(obj, lattice):  # apply lattice modifier to the object
    mod = obj.modifiers.new(name="LatticeMod", type='LATTICE')
    mod.object = lattice
    bpy.context.view_layer.objects.active = obj


def apply_transformation_to_lattice(lattice_obj, transformations):  # apply the transformation to the lattice
    lattice_data = lattice_obj.data
    
    for i, index in enumerate(lattice_activated_vertices):
        point = lattice_data.points[index]
        if i < len(transformations):
            dx, dy, dz = transformations[i]
            point.co_deform.x += dx
            point.co_deform.y += dy
            point.co_deform.z += dz


# def vertex_to_lattice_cell_index(vertex_position, lattice_obj):
#     lattice_matrix_inv = lattice_obj.matrix_world.inverted()
#     local_pos = lattice_matrix_inv @ vertex_position
#     cell_size = np.array([lattice_obj.scale[i] / lattice_obj.data.points_uvw[i] for i in range(3)])
#     cell_index = np.floor((local_pos + lattice_obj.scale) / (2 * cell_size)).astype(int)

#     return tuple(cell_index)


def vertex_to_lattice_cell_index(vertex_position, lattice_obj):  # get the cell index of the vertex in the lattice
    vertex_position = Vector(vertex_position)
    lattice_matrix_inv = lattice_obj.matrix_world.inverted()
    local_pos = lattice_matrix_inv @ vertex_position  # vertex position in the lattice's local coordinate system
    
    cell_size = [
        1 / (getattr(lattice_obj.data, f'points_{dim}') - 1)
        for i, dim in enumerate(['u', 'v', 'w'])
    ]
    local_pos_np = np.array(local_pos)
    clipped_local_pos = np.clip(local_pos_np, -0.4999999, 0.4999999)  # clip the local position to ~ [-0.5, 0.5]

    cell_index = np.floor((clipped_local_pos + np.array([0.5, 0.5, 0.5])) / np.array(cell_size)).astype(int)

    return tuple(cell_index)


##########################################################################################################################################
## EXPERIMENTAL UTILS ##
##########################################################################################################################################
# Gaussian Curvature Utils (Experimental)

def calculate_gaussian_curvature(vertices, faces):
    """
    Calculate the Gaussian curvature using PyMesh.
    
    :param vertices: np.array of shape (N, 3) containing vertex coordinates
    :param faces: np.array of shape (M, 3) containing face indices
    :return: np.array of shape (N,) containing Gaussian curvature for each vertex
    """
    import pymesh
    print(pymesh.__version__)
    mesh = pymesh.form_mesh(vertices, faces)
    mesh.add_attribute("vertex_gaussian_curvature")
    gcur = mesh.get_attribute("vertex_gaussian_curvature")
    print(f"Calculated Gaussian curvature: {gcur}\n")
    return gcur

def compare_gaussian_curvature(vertices1, faces1, vertices2, faces2):
    """
    Compare the Gaussian curvature between two meshes using PyMesh.
    
    :param vertices1, vertices2: np.arrays of shape (N, 3) containing vertex coordinates
    :param faces1, faces2: np.arrays of shape (M, 3) containing face indices
    :return: float, the mean squared difference in Gaussian curvature
    """
    import time
    s = time.time()
    gc1 = calculate_gaussian_curvature(vertices1, faces1)
    gc2 = calculate_gaussian_curvature(vertices2, faces2)
    
    # Ensure the same number of vertices
    assert len(gc1) == len(gc2), "Meshes must have the same number of vertices"
    
    # Calculate mean squared difference
    mse = np.mean((gc1 - gc2)**2)
    e = time.time()
    print(f"Time taken to compare Gaussian curvature of two meshes: {e - s:.2f} seconds")
    return mse


# Overlap Detection Utils (Experimental)

# def check_mesh_overlap(mesh1, mesh2):
#     bm1 = bmesh.new()
#     bm2 = bmesh.new()
#     bm1.from_mesh(mesh1.data)
#     bm2.from_mesh(mesh2.data)
    
#     bm1.transform(mesh1.matrix_world)
#     bm2.transform(mesh2.matrix_world)
    
#     for face1 in bm1.faces:
#         for face2 in bm2.faces:
#             if face1.intersect(face2):
#                 bm1.free()
#                 bm2.free()
#                 return True
                
#     bm1.free()
#     bm2.free()
#     return False


def check_mesh_overlap(mesh1, mesh2):
    bm1 = bmesh.new()
    bm1.from_mesh(mesh1.data)
    bm2 = bmesh.new()
    bm2.from_mesh(mesh2.data)
    bm1.transform(mesh1.matrix_world)
    bm2.transform(mesh2.matrix_world)
    tree1 = BVHTree.FromBMesh(bm1)
    tree2 = BVHTree.FromBMesh(bm2)
    overlap = tree1.overlap(tree2)
    bm1.free()
    bm2.free()
    return len(overlap) > 0

def fix_overlaps(lattice_obj, models):
    """Adjust the lattice to fix overlaps between models."""
    max_iterations = 10
    iteration = 0

    def get_closest_lattice_points(lattice_obj, vertex, threshold=0.2):
        """Get lattice points close to the given vertex."""
        closest_points = []
        for point in lattice_obj.data.points:
            if (Vector(point.co_deform) - Vector(vertex)).length < threshold:
                closest_points.append(point)
        return closest_points

    while iteration < max_iterations:
        overlap_detected = False

        # Check for overlaps between all pairs of models
        for i, model in enumerate(models):
            for j, other_model in enumerate(models):
                if i != j:
                    if check_mesh_overlap(model, other_model):
                        overlap_detected = True
                        # Get overlapping region vertices
                        overlap_vertices = get_overlap_vertices(model, other_model)
                        for vertex in overlap_vertices:
                            # Get lattice points near the overlapping vertex
                            closest_points = get_closest_lattice_points(lattice_obj, vertex)

                            for point in closest_points:
                                # Apply symmetrical adjustment
                                dx = random.uniform(-0.05, 0.05)
                                dy = random.uniform(-0.05, 0.05)
                                dz = random.uniform(-0.05, 0.05)

                                point.co_deform.x += dx
                                point.co_deform.y += dy
                                point.co_deform.z += dz

                                # Find the corresponding symmetrical point and apply the same adjustment
                                sym_point = find_symmetrical_point(lattice_obj, point)
                                if sym_point:
                                    sym_point.co_deform.x += dx
                                    sym_point.co_deform.y += dy
                                    sym_point.co_deform.z += dz

                        # Apply the lattice transform to the model
                        apply_lattice_to_object(model, lattice_obj)
                        apply_lattice_to_object(other_model, lattice_obj)
                        
                        # Update view to reflect changes
                        bpy.context.view_layer.update()

        if not overlap_detected:
            break
        
        iteration += 1

    # Final application of the lattice transform to ensure all adjustments are applied
    for m in models:
        apply_lattice_to_object(m, lattice_obj)
    bpy.context.view_layer.update()


def find_symmetrical_point(lattice_obj, point):
    """Find the point symmetrical to the given point in the lattice."""
    # Assuming symmetry around the origin
    sym_co = Vector(point.co_deform) * -1
    for p in lattice_obj.data.points:
        if (Vector(p.co_deform) - sym_co).length < 0.001:
            return p
    return None


def is_point_inside_mesh(point, mesh_obj):
    """Check if a point is inside a mesh."""
    result, location, normal, index = mesh_obj.closest_point_on_mesh(point)
    return result


def get_overlap_vertices(model, other_model):
    """Get vertices in the overlap region between two models."""
    overlap_vertices = []
    for v in model.data.vertices:
        global_v = model.matrix_world @ v.co
        if is_point_inside_mesh(global_v, other_model):
            overlap_vertices.append(global_v)
    return overlap_vertices



# GA
def evaluate(individual):
    total_distance = 0
    total_curvature_diff = 0
    overlap_penalty = 0
    curvature_weight = 1000

    # Apply transformations to the lattice
    transformations = [(individual[i], individual[i+1], individual[i+2]) for i in range(0, len(individual), 3)]
    apply_transformation_to_lattice(lattice, transformations)
    bpy.context.view_layer.update()

    for i, (start_model, end_model) in enumerate(zip(matched_start_models, matched_end_models)):
        reset_lattice(lattice)
        apply_transformation_to_lattice(lattice, transformations)
        bpy.context.view_layer.update()

        transformed_vertices = get_transformed_vertex_positions(start_model)
        target_vertices = end_model_vertices[i]

        def get_faces_as_array(obj):
            faces = []
            for poly in obj.data.polygons:
                if len(poly.vertices) == 3:
                    faces.append([poly.vertices[0], poly.vertices[1], poly.vertices[2]])
                elif len(poly.vertices) > 3:
                    for i in range(1, len(poly.vertices) - 1):
                        faces.append([poly.vertices[0], poly.vertices[i], poly.vertices[i+1]])
            return np.array(faces, dtype=np.int32)

        start_faces = get_faces_as_array(start_model)
        end_faces = get_faces_as_array(end_model)
        
        try:
            curvature_diff = compare_gaussian_curvature(
                transformed_vertices, start_faces,
                target_vertices, end_faces
            )
        except Exception as e:
            curvature_diff = 0

        selected_indices_for_model = selected_indices[i] 
        size_weight = normalized_start_model_sizes[start_model.name]
        distance = calculate_total_distance2(transformed_vertices, target_vertices, selected_indices_for_model) * size_weight
        total_distance += distance

        total_curvature_diff += curvature_diff

        for j, other_model in enumerate(matched_start_models):
            if i != j:
                if check_mesh_overlap(start_model, other_model):
                    overlap_penalty += 100  # Arbitrary penalty value for overlap

    fitness = total_distance + total_curvature_diff * curvature_weight + overlap_penalty
    return fitness,



clear_lattice_modifier()

lattice_location = (0, 0, 0.8)  # location of the lattice origin
lattice_scale = (1.6, 1.6, 1.6)
subdivisions = 15  # 15x15x15 lattice

lattice_name = "OptimizedLattice"
lattice = create_and_setup_lattice(lattice_name, lattice_location, lattice_scale, subdivisions)

# start_collection_group_name = "MOE"
start_collection_group_name = "MOE-DEFAULT-AVATAR"  # change this to "MOE" if necessary
end_collection_group_name = "MANUKA"
pre_guided_lattice_name = "MOE2MANUKA-ROUGH"

start_collection_group = bpy.data.collections.get(start_collection_group_name)  # collection of the start models
end_collection_group = bpy.data.collections.get(end_collection_group_name)  # collection of the end models

start_collection_models = get_sorted_models_from_collection_group(start_collection_group)  # format: {collection_name: [models], }
end_collection_models = get_sorted_models_from_collection_group(end_collection_group)

matched_start_models = []
matched_end_models = []

# Find matching start and end models
for collection_name, start_models in start_collection_models.items():
    # end models with the same collection name
    if collection_name in end_collection_models:
        end_models = end_collection_models[collection_name]
        # number of end models with the same number of vertices, format: {vertex_count: end_model_count, }
        mesh_count_to_model_count = Counter(len(model.data.vertices) for model in end_models)

        for start_model in start_models:
            for end_model in end_models:
                # if number of vertices of start model is the same as end model and only one end model has this number of vertices
                if (len(start_model.data.vertices) == len(end_model.data.vertices) and 
                        mesh_count_to_model_count[len(end_model.data.vertices)] == 1):
                    matched_start_models.append(start_model)
                    matched_end_models.append(end_model)
                    
print ("Listed matched start models")                    
print (matched_start_models)
print ("Listed matched end models (targets)")
print (matched_end_models)

start_model_vertices = []
end_model_vertices = []
selected_indices = []
start_model_sizes = {}
cell_vertex_count = {}

pre_guided_lattice_name = "MOE2MANUKA-ROUGH"
pre_guided_lattice = bpy.data.objects.get(pre_guided_lattice_name)
if pre_guided_lattice is None:
    print(f"Lattice '{pre_guided_lattice_name}' not found.")

for start_model in matched_start_models:
    if pre_guided_lattice is not None:    
        apply_lattice_to_object(start_model, pre_guided_lattice)  # apply "MOE2MANUKA-ROUGH" to the start model
    vertices = get_transformed_vertex_positions(start_model)
    start_model_vertices.append(vertices)    
    apply_lattice_to_object(start_model, lattice)  # apply "OptimizedLattice" to the start model
    # select 10,000 vertices if the number of vertices is more than 1100, otherwise select all vertices
    if len(vertices) >= 1100:
        selected_indices.append( 
            random.sample(range(len(vertices)), min(10000, len(vertices)))
        )
    # if the number of vertices is less than 1100, select all vertices
    else:
        selected_indices.append(range(len(vertices)))
        
    size = start_model.dimensions.x * start_model.dimensions.y * start_model.dimensions.z  # volume of the start model
    start_model_sizes[start_model.name] = size
    for vertex in vertices:
        cell_index = vertex_to_lattice_cell_index(tuple(vertex), lattice)  # get the cell index of the vertex, format: (x, y, z)
        if cell_index in cell_vertex_count:
            cell_vertex_count[cell_index] += 1
        else:
            cell_vertex_count[cell_index] = 1


unique_cell_indices = list(cell_vertex_count.keys())
unique_cell_vertices = {}  # format: {vertex_index (format: int): num_of_vertices (format: int), }
N = subdivisions  # number of points in each dimension of the lattice
for cell_index in unique_cell_indices:  # cells that contain at least one vertex of the start models
    x = cell_index[0]
    y = cell_index[1]
    z = cell_index[2]
    indices = [  # indices of the 8 vertices of the cell. In case of (0, 0, 0), the indices are [0, 1, N, N+1, N*N, N*N+1, N*N+N, N*N+N+1]
        x+y*N+z*N*N,
        x+y*N + z*N*N + 1,
        x+(y+1)*N + z*N*N,
        x+(y+1)*N + z*N*N + 1,
        x+y*N + (z+1)*N*N,
        x+y*N + (z+1)*N*N + 1,
        x+(y+1)*N + (z+1)*N*N,
        x+(y+1)*N + (z+1)*N*N + 1
    ]
    # count the number of vertices with the same vertex index
    for idx in indices:
        if idx in unique_cell_vertices:
            unique_cell_vertices[idx] += 1
        else:
            unique_cell_vertices[idx] = 1

#lattice_activated_vertices = sorted(list(unique_cell_vertices.keys()))
#lattice_activated_vertices = sorted([cell_index for cell_index, count in unique_cell_vertices.items() if count > 1])
# lattice_activated_vertices = sorted([cell_index for cell_index, count in unique_cell_vertices.items() if count > 0])
lattice_activated_vertices = sorted(list(unique_cell_vertices.keys()))

# count is always at least 1, so no need to check for count > 0

log_start_model_sizes = {model_name: math.log1p(size) for model_name, size in start_model_sizes.items()}
# log(1 + size), where size is the volume of each start model

min_log_size = min(log_start_model_sizes.values())
max_log_size = max(log_start_model_sizes.values())

normalized_start_model_sizes = {}

# Normalize the log sizes of the start models
for model_name, log_size in log_start_model_sizes.items():
    if max_log_size - min_log_size > 0:
        normalized_size = (log_size - min_log_size) / (max_log_size - min_log_size)
    else:
        normalized_size = 1 
    normalized_start_model_sizes[model_name] = normalized_size


# Get the transformed vertices of the end models
for end_model in matched_end_models:
    end_model_vertices.append(get_transformed_vertex_positions(end_model))
    
print(f"# of start models : {len(start_model_vertices)}")
print(f"# of end models : {len(end_model_vertices)}")
print(f"Dimension of chromosome : {len(lattice_activated_vertices) * 3}")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -0.1/subdivisions, 0.1/subdivisions)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(lattice_activated_vertices) * 3) 
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

NGEN = 10
population = toolbox.population(n=10) 

print("Start")
for gen in range(NGEN):
    gen_start_time = time.time()
    
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    lengths = [ind.fitness.values[0] for ind in population]
    mean = sum(lengths) / len(population)
    minimum = min(lengths)

    gen_end_time = time.time()
    gen_time = gen_end_time - gen_start_time
    print(f"세대 {gen+1}/{NGEN} - 최소 적합도: {minimum}, 평균 적합도: {mean}, 계산 시간: {gen_time:.2f}초")
    
    elapsed_time = gen_end_time - start_time
    estimated_total_time = (elapsed_time / (gen + 1)) * NGEN
    remaining_time = estimated_total_time - elapsed_time
    print(f"예상 남은 시간: {remaining_time:.2f}초")
    
best_individual = tools.selBest(population, 1)[0]

print("Final distance")
print(evaluate(best_individual))
transformations = [(best_individual[i*3], best_individual[i*3+1], best_individual[i*3+2]) for i in range(len(lattice_activated_vertices))]
reset_lattice(lattice)
apply_transformation_to_lattice(lattice, transformations)
print("최적의 개체의 적합도:", best_individual.fitness.values[0])


## EXPERIMENTAL ##

bpy.context.view_layer.update()

print("Best individual transformations applied to lattice:", transformations)

# Fix overlaps after applying the best transformation
fix_overlaps(lattice, matched_start_models + matched_end_models)
print("Overlaps fixed.")

## EXPERIMENTAL ##


end_time = time.time()  # 프로그램 실행 종료 시간
total_time = end_time - start_time
print(f"프로그램 총 실행 시간: {total_time}초")