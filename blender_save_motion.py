import bpy
import mathutils
import json

# Load your FBX files
fbx_file_paths = [
    "/home/jianrenw/Downloads/studio-mocap-emotes/thanks-gentleman-bow.fbx",
    "/home/jianrenw/Downloads/studio-mocap-emotes/thanks-bow.fbx",
    "/home/jianrenw/Downloads/studio-mocap-emotes/greeting-small-wave.fbx",
    "/home/jianrenw/Downloads/studio-mocap-emotes/directing-over-there.fbx",
    "/home/jianrenw/Downloads/studio-mocap-evolution-of-dance-vol-1/billiejean_dance01.fbx",
]

def clear_scene():
    """Clear all objects in the current Blender scene without resetting Blender's state."""
    bpy.ops.object.select_all(action='SELECT')  # Select all objects
    bpy.ops.object.delete()  # Delete all selected objects
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def select_object(obj):
    """Ensure the object is selected and active."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

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

    # Ensure the armature is selected and active
    for obj in armature_objects:
        select_object(obj)
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

def get_last_keyframe():
    """Find the last keyframe from all armatures in the scene."""
    armature_objects = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']

    if not armature_objects:
        print("No armature found in the scene.")
        return None, None

    last_keyframe = 0

    for obj in armature_objects:
        # Ensure the armature is selected and active
        select_object(obj)

        # Look through all keyframe points in F-Curves (animation data)
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            for fcurve in action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    last_keyframe = max(last_keyframe, keyframe.co[0])  # `co[0]` is the frame number
    
    return int(last_keyframe)

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

# Process each FBX file separately
for fbx_file_path in fbx_file_paths:
    # Clear the scene before loading each new FBX file
    clear_scene()

    # Import the FBX file
    bpy.ops.import_scene.fbx(filepath=fbx_file_path)

    # Find the last keyframe from the FBX animation
    last_keyframe = get_last_keyframe()

    if last_keyframe is not None and last_keyframe > 0:
        start_frame = 1  # Default start frame (could also be the first keyframe)
        end_frame = last_keyframe

        # Extract the motion sequence using the actual animation length
        motion_data = extract_motion_sequence(start_frame, end_frame)

        # Specify the output JSON file path
        output_file_path = "/home/jianrenw/mocap/data/actioncore/raw/{}.json".format(fbx_file_path.split("/")[-1].split(".")[0])
        
        # Save the motion sequence to a JSON file
        if motion_data:
            save_to_json(motion_data, output_file_path)
        else:
            print("No motion data found.")
    else:
        print("No keyframes found in the FBX file.")
