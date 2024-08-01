import pymesh
import numpy as np

def create_cube(center, size):
    """Create a cube mesh centered at a given position with a given size."""
    half_size = size / 2.0
    vertices = np.array([
        center + [-half_size, -half_size, -half_size],
        center + [ half_size, -half_size, -half_size],
        center + [ half_size,  half_size, -half_size],
        center + [-half_size,  half_size, -half_size],
        center + [-half_size, -half_size,  half_size],
        center + [ half_size, -half_size,  half_size],
        center + [ half_size,  half_size,  half_size],
        center + [-half_size,  half_size,  half_size],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 3, 7], [0, 7, 4],
    ])
    return pymesh.form_mesh(vertices, faces)

def detect_overlap(mesh1, mesh2):
    """Detect if two meshes overlap."""
    intersection = pymesh.boolean(mesh1, mesh2, operation="intersection")
    return intersection.num_vertices > 0

def fix_overlap(mesh1, mesh2, direction=np.array([1, 0, 0]), distance=0.1):
    """Fix overlapping parts by moving mesh2 in the specified direction."""
    while detect_overlap(mesh1, mesh2):
        # Make a copy of the vertices to modify them
        vertices_copy = np.copy(mesh2.vertices)
        vertices_copy += direction * distance
        mesh2 = pymesh.form_mesh(vertices_copy, mesh2.faces)
    return mesh2

def main():
    # Create two overlapping cubes
    cube1 = create_cube(center=np.array([0, 0, 0]), size=2.0)
    cube2 = create_cube(center=np.array([1, 1, 1]), size=2.0)

    # Save the original cubes for reference
    pymesh.save_mesh("cube1.obj", cube1)
    pymesh.save_mesh("cube2.obj", cube2)

    # Detect overlap
    if detect_overlap(cube1, cube2):
        print("Meshes overlap. Fixing overlap...")
        cube2 = fix_overlap(cube1, cube2)
        print("Overlap fixed.")
    else:
        print("Meshes do not overlap.")

    # Save the fixed mesh2
    pymesh.save_mesh("fixed_cube2.obj", cube2)

if __name__ == "__main__":
    main()
