import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

cv2.namedWindow("preview")
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

def mse(img1, img2):
    """Calculate the Mean Squared Error between two images."""
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse

def testWebcam(requested_width, requested_height, duration=30):
    """Test the webcam at a specific resolution for a given duration."""
    print(f"Testing {requested_width} x {requested_height} for {duration} seconds")
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

    if not vc.isOpened():
        print("Error: Unable to open webcam.")
        return None

    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # Ensure FPS is valid
        print("Error: Unable to determine FPS. Defaulting to 30.")
        fps = 30
    wait_time = int(1000.0 / float(fps))  # Safe calculation after validation
    print(f"Actual resolution: {width} x {height}, FPS: {fps:.2f}")

    if (requested_width != width) or (requested_height != height):
        print("Warning: Unable to support requested width and height.")
        vc.release()
        return None

    frame_count = 0
    bad_frame_count = 0
    refocus_count = 0
    prev_frame = None
    start_time = time.time()
    frame_timestamps = []  # Track timestamps of frame captures

    while time.time() - start_time < duration:
        rval, frame = vc.read()
        if not rval:
            print("Warning: Failed to read frame.")
            frame_timestamps.append((time.time() - start_time, 0))  # Frame dropped
            continue

        frame_count += 1
        frame_timestamps.append((time.time() - start_time, 1))  # Frame successfully captured
        if prev_frame is not None:
            gray_img1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            error = mse(gray_img1, gray_img2)
            if error > 10.0:
                bad_frame_count += 1
            elif error > 1.0:
                refocus_count += 1

        remaining_frames = max(0, int(fps * duration) - frame_count)
        #cv2.putText(frame, f"{width} x {height} @ {fps:.2f} Remaining: {remaining_frames}", org, font, fontScale, color, thickness)
        #cv2.putText(frame, f"Bad Frames: {bad_frame_count}, Refocuses: {refocus_count}", (org[0], org[1] + 30), font, fontScale, color, thickness)
        cv2.imshow("preview", frame)

        prev_frame = frame
        key = cv2.waitKey(wait_time)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")

    expected_frames = int(fps * duration)
    dropped_frames = expected_frames - frame_count
    print(f"Out of {expected_frames} expected frames, captured {frame_count}, dropped {dropped_frames}")
    print(f"Bad frames: {bad_frame_count}, Refocuses: {refocus_count}")

    return {
        "resolution": f"{width} x {height}",
        "captured_frames": frame_count,
        "dropped_frames": dropped_frames,
        "bad_frames": bad_frame_count,
        "refocuses": refocus_count,
        "frame_timestamps": frame_timestamps
    }

def plot_frame_status(results):
    """Plot the frame success vs. drop timeline for each resolution using timestamps."""
    for result in results:
        if "frame_timestamps" not in result:
            continue
        frame_timestamps = result["frame_timestamps"]
        resolution = result["resolution"]

        times = [timestamp[0] for timestamp in frame_timestamps]  # Extract timestamps for x-axis
        statuses = [timestamp[1] for timestamp in frame_timestamps]  # Extract statuses for y-axis

        plt.figure()
        plt.step(times, statuses, where="post", label="Frame Status (1=Success, 0=Drop)", color="blue")
        plt.title(f"Frame Timeline for {resolution}")
        plt.xlabel("Time (seconds)")  # Correctly label x-axis as time
        plt.ylabel("Frame Status")  # Correctly label y-axis as frame status
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_frame_timestamps(results):
    """Plot the deltas between frame captures for each resolution."""
    for result in results:
        if "frame_timestamps" not in result:
            continue
        frame_timestamps = result["frame_timestamps"]
        resolution = result["resolution"]

        # Extract timestamps for successful frames (status = 1)
        capture_times = [timestamp[0] for timestamp in frame_timestamps if timestamp[1] == 1]

        # Calculate deltas (time differences) between consecutive timestamps
        deltas = [capture_times[i] - capture_times[i - 1] for i in range(1, len(capture_times))]

        plt.figure()
        # Commented out frame capture times
        # plt.scatter(capture_times, [1] * len(capture_times), label="Frame Captures", color="blue", marker="o")
        if deltas:
            plt.plot(capture_times[1:], deltas, label="Delta Between Frames", color="red")  # Solid red line
        plt.title(f"Frame Capture Deltas for {resolution}")
        plt.xlabel("Time (seconds)")  # X-axis shows time
        plt.ylabel("Delta Time (seconds)")  # Updated y-axis label
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add gridlines
        plt.show()

def testWebcamLiveChart(requested_width, requested_height, duration=30):
    """Test the webcam at a specific resolution for a given duration with a live chart."""
    print(f"Testing {requested_width} x {requested_height} for {duration} seconds")
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

    if not vc.isOpened():
        print("Error: Unable to open webcam.")
        return None

    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # Ensure FPS is valid
        print("Error: Unable to determine FPS. Defaulting to 30.")
        fps = 30
    wait_time = int(1000.0 / float(fps))  # Safe calculation after validation
    print(f"Actual resolution: {width} x {height}, FPS: {fps:.2f}")

    if (requested_width != width) or (requested_height != height):
        print("Warning: Unable to support requested width and height.")
        vc.release()
        return None

    frame_count = 0
    start_time = time.time()
    capture_times = []  # Track timestamps of successful frame captures
    deltas = []  # Track deltas between consecutive frame captures

    # Setup live chart
    fig, ax = plt.subplots()
    ax.set_title(f"Live Frame Capture Deltas for {width} x {height}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Delta Time (seconds)")
    line, = ax.plot([], [], color="red", label="Delta Between Frames")
    ax.legend()
    ax.grid(True)

    def update_chart(frame):
        """Update the live chart with new data."""
        if len(capture_times) > 1:
            line.set_data(capture_times[1:], deltas)
            ax.set_xlim(0, max(capture_times))
            ax.set_ylim(0, max(deltas) if deltas else 1)
        return line,

    ani = FuncAnimation(fig, update_chart, interval=500, cache_frame_data=False)  # Disable frame data caching

    # Run the webcam test in a separate thread
    def webcam2_test():
        nonlocal frame_count, capture_times, deltas
        prev_time = None
        timeout_start = time.time()
        timeout_duration = 5  # Timeout duration in seconds

        while time.time() - start_time < duration:
            rval, frame = vc.read()
            if not rval:
                print("Warning: Failed to read frame.")
                if time.time() - timeout_start > timeout_duration:
                    print("Error: Frame capture timeout exceeded.")
                    break
                continue

            timeout_start = time.time()  # Reset timeout on successful frame capture
            frame_count += 1
            current_time = time.time() - start_time
            capture_times.append(current_time)

            if prev_time is not None:
                deltas.append(current_time - prev_time)

            prev_time = current_time

            #print(f"Frame {frame_count} captured at {current_time:.2f} seconds")
            #cv2.imshow("preview", frame)
            #print(f"Frame {frame_count} displayed at {current_time:.2f} seconds")

            key = cv2.waitKey(wait_time) & 0xFF  # Ensure proper key handling
            if key == 27:  # exit on ESC
                break
            #print("Waiting for key press...")

        vc.release()
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
        

    webcam_thread = threading.Thread(target=webcam2_test)
    webcam_thread.start()

    # Show the live chart
    plt.show()

    # Wait for the webcam thread to finish
    while webcam_thread.is_alive():
        print("Waiting for webcam thread to finish... {time.time() - start_time:.2f} seconds elapsed")
        time.sleep(60)

    webcam_thread.join()

    return {
        "resolution": f"{width} x {height}",
        "captured_frames": frame_count,
        "deltas": deltas
    }

def main():
    """Main function to test multiple resolutions with live charts."""
    resolutions = [
        (640, 480),
        (1280, 720)
    ]
    results = []

    for width, height in resolutions:
        print(f"Starting test for resolution {width} x {height}")
        result = testWebcamLiveChart(width, height, 10*60)
        if result:
            results.append(result)
            print(f"Resolution {result['resolution']}: SUCCESS, Frames Captured: {result['captured_frames']}")
        else:
            print(f"Resolution {width} x {height}: FAILED")

if __name__ == "__main__":
    main()
