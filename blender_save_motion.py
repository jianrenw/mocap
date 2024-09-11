import bpy
import mathutils
import json

# Load your FBX file
fbx_file_path = "/home/jianrenw/Downloads/studio-mocap-emotes/thanks-gentleman-bow.fbx"
bpy.ops.import_scene.fbx(filepath=fbx_file_path)

def get_bone_world_matrix(obj, bone_name):
    """Get the bone's world matrix."""
    bone = obj.pose.bones[bone_name]
    # Multiply object matrix with bone matrix to get the global transform
    return obj.matrix_world @ bone.matrix

def extract_keypoint_poses_for_frame(frame_number):
    """Extract keypoint poses for a specific frame."""
    bpy.context.scene.frame_set(frame_number)  # Set the scene to the desired frame
    joint_poses = {}

    # Check if there are any armature objects in the scene
    armature_objects = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    if not armature_objects:
        print("No armature found in the scene. Ensure the FBX contains an armature object.")
        return joint_poses

    # Process each armature in the scene
    for obj in armature_objects:
        for bone in obj.pose.bones:
            bone_world_matrix = get_bone_world_matrix(obj, bone.name)
            translation = bone_world_matrix.translation
            rotation = bone_world_matrix.to_euler()
            scaling = bone_world_matrix.to_scale()
            
            joint_poses[bone.name] = {
                'translation': (translation.x, translation.y, translation.z),
                'rotation': (rotation.x, rotation.y, rotation.z),
                'scaling': (scaling.x, scaling.y, scaling.z)
            }
    
    return joint_poses

def extract_motion_sequence(start_frame, end_frame):
    """Extract keypoint poses for a sequence of frames."""
    motion_data = {}

    # Loop through each frame and capture the joint poses
    for frame in range(start_frame, end_frame + 1):
        print(f"Extracting frame {frame}...")
        motion_data[frame] = extract_keypoint_poses_for_frame(frame)
    
    return motion_data

def save_to_json(data, file_path):
    """Save the motion sequence data to a JSON file."""
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Motion data saved to {file_path}")

# Specify the start and end frames (you can adjust this to match your motion sequence)
start_frame = bpy.context.scene.frame_start
end_frame = bpy.context.scene.frame_end

# Extract the motion sequence
motion_data = extract_motion_sequence(start_frame, end_frame)

# Specify the output JSON file path
output_file_path = "/home/jianrenw/mocap/data/actioncore/raw/thanks-gentleman-bow.json"

# Save the motion sequence to a JSON file
if motion_data:
    save_to_json(motion_data, output_file_path)
else:
    print("No motion data found.")