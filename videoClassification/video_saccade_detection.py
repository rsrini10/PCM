import numpy as np
import cv2
import pandas as pd
import imageio

def detect_saccades_from_stimulus(stimulus, threshold_factor=3):
    """
    Detects saccades from the stimulus signal using the derivative method.
    
    Parameters:
        stimulus (np.ndarray): The stimulus signal.
        threshold_factor (float): Multiplier for standard deviation to set detection threshold.
    
    Returns:
        np.ndarray: Indices of detected saccades in the stimulus (in frame units).
    """
    stimulus_gradient = np.diff(stimulus)
    stimulus_threshold = np.std(stimulus_gradient) * threshold_factor
    stimulus_saccades = np.where(np.abs(stimulus_gradient) > stimulus_threshold)[0]
    return stimulus_saccades


def detect_initial_saccades_from_video(video_path, fps, stimulus_saccades, threshold_factor=3):
    """
    Performs initial saccade detection from the video using optical flow.
    
    Parameters:
        video_path (str): Path to the video file.
        fps (int): Frames per second of the video.
        stimulus_saccades (array): Indices of saccades detected in the stimulus.
        threshold_factor (float): Multiplier for standard deviation to set detection threshold.
    
    Returns:
        np.ndarray: Initial detected saccades in the video (frame indices).
    """
    vid = imageio.get_reader(video_path, 'ffmpeg')
    prev_gray = None
    horizontal_motion_magnitudes = []
    frame_indices = []

    for i, frame in enumerate(vid):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            horizontal_motion = np.abs(flow[..., 0]).mean()
            horizontal_motion_magnitudes.append(horizontal_motion)
            frame_indices.append(i)
        prev_gray = gray

    horizontal_motion_magnitudes = np.array(horizontal_motion_magnitudes)
    frame_indices = np.array(frame_indices)
    horizontal_threshold = np.std(horizontal_motion_magnitudes) * threshold_factor

    expected_delay = int(0.1 * fps)  # Adjust delay (~100ms)
    predicted_saccades = stimulus_saccades + expected_delay

    saccades = []
    for pred_saccade in predicted_saccades:
        search_range = np.arange(max(0, int(pred_saccade) - 10), min(len(horizontal_motion_magnitudes), int(pred_saccade) + 10))
        if len(search_range) > 0:
            best_match = search_range[np.argmax(horizontal_motion_magnitudes[search_range])]
            if horizontal_motion_magnitudes[best_match] > horizontal_threshold:
                saccades.append(best_match)

    return np.array(saccades)


def refine_video_saccade_detection(video_path, fps, stimulus_saccades, max_iterations=10):
    """
    Refines video-based saccade detection to ensure the same number of saccades as in the stimulus.
    
    Parameters:
        video_path (str): Path to the video file.
        fps (int): Frames per second of the video.
        stimulus_saccades (array): Indices of saccades detected in the stimulus.
        max_iterations (int): Maximum number of iterations for refining threshold.

    Returns:
        np.ndarray: Refined detected saccades in the video (frame indices).
    """
    threshold_factor = 3.0

    for iteration in range(max_iterations):
        detected_saccades = detect_initial_saccades_from_video(video_path, fps, stimulus_saccades, threshold_factor)
        num_stimulus_saccades = len(stimulus_saccades)
        num_detected_saccades = len(detected_saccades)

        if num_detected_saccades == num_stimulus_saccades:
            return detected_saccades  # Accept result

        elif num_detected_saccades > num_stimulus_saccades:
            threshold_factor *= 1.2  # Stricter threshold to reduce detections

        elif num_detected_saccades == 0:
            threshold_factor /= 1.5  # Loosen threshold until at least one is detected

        else:
            detected_saccades_times = detected_saccades / fps
            predicted_saccades_times = stimulus_saccades / fps

            delays = []
            matched_saccades = []
            for s in detected_saccades_times:
                closest_stimulus_saccade = min(predicted_saccades_times, key=lambda x: abs(x - s))
                delay = s - closest_stimulus_saccade
                delays.append(delay)
                matched_saccades.append(closest_stimulus_saccade)

            avg_delay = np.mean(delays)
            missing_saccades = set(predicted_saccades_times) - set(matched_saccades)
            for ms in missing_saccades:
                detected_saccades = np.append(detected_saccades, int((ms + avg_delay) * fps))

            return np.sort(detected_saccades).astype(int)

    raise RuntimeError("Failed to refine threshold after 10 iterations.")


def detect_saccades_from_video(video_path, stimulus, fps):
    """
    Main function to detect saccades in a video given the stimulus as prior knowledge.
    
    Parameters:
        video_path (str): Path to the video file.
        stimulus (np.ndarray): The stimulus signal.
        fps (int): Frames per second of the video.
    
    Returns:
        np.ndarray: Binary vector of detected saccades in the video (same length as video frames).
    """
    stimulus_saccades = detect_saccades_from_stimulus(stimulus)
    refined_saccades = refine_video_saccade_detection(video_path, fps, stimulus_saccades)

    saccade_vector = np.zeros(int(fps * len(stimulus) / fps))  # Initialize binary vector
    saccade_vector[refined_saccades] = 1

    return saccade_vector

def detect_saccades_from_video_simple(video_path, stimulus, fps):
    """
    Main function to detect saccades in a video given the stimulus as prior knowledge.
    NOTE: This version don't use complex refinement since that's too slow. Ramdomly choose a latency value.
    
    Parameters:
        video_path (str): Path to the video file.
        stimulus (np.ndarray): The stimulus signal.
        fps (int): Frames per second of the video.
    
    Returns:
        np.ndarray: Binary vector of detected saccades in the video (same length as video frames).
    """
    stimulus_saccades = detect_saccades_from_stimulus(stimulus)
    refined_saccades = stimulus_saccades + np.random.randint(5, 15)
    # refined_saccades = refine_video_saccade_detection(video_path, fps, stimulus_saccades)

    saccade_vector = np.zeros(int(fps * len(stimulus) / fps))  # Initialize binary vector
    saccade_vector[refined_saccades] = 1

    return saccade_vector


if __name__ == "__main__":
    newest_video_path="/Users/tianyulin/My/JHU/Courses/PCM/clinicalData/2023_7_12_clip5/0b97d7f0-e54b-4dea-9734-adbeec46da69_2/saccade.avi"
    stimulus_newest="/Users/tianyulin/My/JHU/Courses/PCM/clinicalData/2023_7_12_clip5/0b97d7f0-e54b-4dea-9734-adbeec46da69_2/saccade.csv"
    df = pd.read_csv(stimulus_newest)  # Replace with your file
    stimulus = df["stimulus"].values  # Extract the stimulus column
    waveform = df["waveform"].values

    fps=60

    # Example Usage (Assuming video_path and stimulus are provided)
    """ API: video and stimulus to get estimated saccade position"""
    saccade_vector_video_final = detect_saccades_from_video(newest_video_path, stimulus, fps)
    video_saccade_indices = np.arange(len(stimulus))[saccade_vector_video_final==1]

    # Display the first few values of the final saccade detection result
    print(saccade_vector_video_final, video_saccade_indices)  # First 50 frames as an example output

    import matplotlib.pyplot as plt
    plt.plot(stimulus)
    plt.plot(waveform)
    # plt.scatter(video_saccade_indices, stimulus[video_saccade_indices])
    plt.scatter(video_saccade_indices, waveform[video_saccade_indices])
    plt.show()