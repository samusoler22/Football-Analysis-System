import pickle
import os

def init_track_stubs():
    # Initialize empty tracks dictionary
    tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }
    
    with open(os.path.join("stubs", "track_stubs.pkl"), "wb") as f:
        pickle.dump(tracks, f)

def init_camera_movement_stub():
    # Initialize empty camera movement list
    camera_movement = []
    
    with open(os.path.join("stubs", "camera_movement_stub.pkl"), "wb") as f:
        pickle.dump(camera_movement, f)

if __name__ == "__main__":
    # Create stubs directory if it doesn't exist
    os.makedirs("stubs", exist_ok=True)
    
    # Initialize stub files
    init_track_stubs()
    init_camera_movement_stub()
    
    print("Stub files initialized successfully!") 