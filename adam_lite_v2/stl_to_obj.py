import open3d as o3d
import os
import copy

mesh_dir = '/home/jianrenw/mocap/adam_lite_v2/meshes'
mesh_files = os.listdir(mesh_dir)
target_num_triangles = 20000
for mesh_file in mesh_files:
    if mesh_file.endswith('.STL'):
        # Load the STL file
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, mesh_file))
        original_mesh = copy.deepcopy(mesh)
        mesh = mesh.simplify_quadric_decimation(target_num_triangles)
        mesh.remove_unreferenced_vertices()
        # check watertightness
        if mesh.is_watertight():
            print(mesh_file, "Downsampled mesh is watertight.")
        else:
            print("Downsampled mesh is not watertight. Attempting to repair...")
            mesh = original_mesh.remove_degenerate_triangles()
            mesh = mesh.simplify_quadric_decimation(target_num_triangles)
            mesh.remove_unreferenced_vertices()
            if mesh.is_watertight():
                print(mesh_file, "Mesh repaired and is now watertight.")
            else:
                print(mesh_file, "Failed to repair the mesh.")

        # Export the mesh to an OBJ file
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, mesh_file[:-4] + '.obj'), mesh)