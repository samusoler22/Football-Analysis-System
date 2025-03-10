from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import cv2
import time
from datetime import datetime, timedelta

def process_video_chunk(video_path, start_frame, chunk_size):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    count = 0
    
    while count < chunk_size:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    
    cap.release()
    return frames, count

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def main():
    print("\nStarting video processing...")
    start_time = time.time()
    
    video_path = r"input_videos\SPALDING UNITED-HARBOROUGH TOWN.mp4"
    chunk_size = 100  # Process 100 frames at a time
    
    # Get total frame count
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Changed to FRAME_COUNT
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\nVideo details:")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Estimated duration: {format_time(total_frames/fps)}\n")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_videos/output.avi", fourcc, fps, (frame_width, frame_height))

    # Initialize tracker and other components
    print("Initializing components...")
    tracker = Tracker("models/best.pt")
    camera_movement_estimator = None
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()

    all_tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }
    team_ball_control = []
    camera_movement_per_frame = []

    # Process video in chunks
    processed_frames = 0
    chunk_times = []
    
    print("\nProcessing video...")
    for start_frame in range(0, total_frames, chunk_size):
        chunk_start_time = time.time()
        end_frame = min(start_frame + chunk_size, total_frames)
        
        # Calculate progress
        progress = (start_frame / total_frames) * 100
        
        # Calculate estimated time remaining
        if chunk_times:
            avg_time_per_chunk = sum(chunk_times) / len(chunk_times)
            chunks_remaining = (total_frames - start_frame) / chunk_size
            eta = avg_time_per_chunk * chunks_remaining
            eta_str = format_time(eta)
            
            # Calculate processing speed
            frames_per_second = chunk_size / avg_time_per_chunk
            speed_str = f"Speed: {frames_per_second:.1f} fps"
        else:
            eta_str = "calculating..."
            speed_str = ""

        print(f"\rProgress: {progress:.1f}% (Frame {start_frame}/{total_frames}) - ETA: {eta_str} - {speed_str}", end="", flush=True)
        
        # Read chunk of frames
        video_frames, actual_frames = process_video_chunk(video_path, start_frame, chunk_size)
        if not video_frames:
            break

        processed_frames += len(video_frames)

        # Initialize camera movement estimator with first frame if not done yet
        if camera_movement_estimator is None:
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            # Initialize team colors with first frame
            tracks = tracker.get_object_tracks(video_frames[:1], read_from_stub=False)
            if tracks['players'] and tracks['players'][0]:
                team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

        # Process chunk
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
        tracker.add_position_to_tracks(tracks)

        # Camera movement estimation
        chunk_camera_movement = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=False)
        camera_movement_per_frame.extend(chunk_camera_movement)
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, chunk_camera_movement)

        # View transformation
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Ball position interpolation
        if tracks['ball']:
            tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

        # Speed and distance estimation
        speed_and_distance_estimator.add_speed_and_sistance_to_tracks(tracks)

        # Team assignment and ball control
        chunk_team_control = []
        for frame_num, player_tracks in enumerate(tracks['players']):
            for player_id, track in player_tracks.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]

            if tracks['ball'] and frame_num < len(tracks['ball']):
                ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', None)
                if ball_bbox is not None:
                    assigned_player = player_assigner.assign_ball_to_player(player_tracks, ball_bbox)
                    if assigned_player != -1:
                        tracks['players'][frame_num][assigned_player]['has_ball'] = True
                        chunk_team_control.append(tracks['players'][frame_num][assigned_player]['team'])
                    elif chunk_team_control:
                        chunk_team_control.append(chunk_team_control[-1])
                    else:
                        chunk_team_control.append(1)  # Default to team 1 if no control determined

        team_ball_control.extend(chunk_team_control)

        # Draw annotations and save frames
        output_frames = tracker.draw_annotations(video_frames, tracks, np.array(team_ball_control[-len(video_frames):]))
        output_frames = camera_movement_estimator.draw_camera_movement(output_frames, chunk_camera_movement)
        output_frames = speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

        # Write frames to video
        for frame in output_frames:
            out.write(frame)

        # Extend all_tracks with current chunk's tracks
        for key in all_tracks:
            all_tracks[key].extend(tracks[key])

        # Record chunk processing time
        chunk_time = time.time() - chunk_start_time
        chunk_times.append(chunk_time)

    # Release video writer
    out.release()

    # Save final tracks and camera movement to stubs
    print("\n\nSaving tracking data...")
    import pickle
    with open("stubs/track_stubs.pkl", "wb") as f:
        pickle.dump(all_tracks, f)
    with open("stubs/camera_movement_stub.pkl", "wb") as f:
        pickle.dump(camera_movement_per_frame, f)

    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total processing time: {format_time(total_time)}")
    print(f"Average processing speed: {processed_frames/total_time:.1f} frames per second")
    print(f"Output video saved to: output_videos/output.avi")

if __name__ == '__main__':
    main()