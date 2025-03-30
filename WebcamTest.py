import cv2
import numpy as np
import time

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

    while time.time() - start_time < duration:
        rval, frame = vc.read()
        if not rval:
            print("Warning: Failed to read frame.")
            continue

        frame_count += 1
        if prev_frame is not None:
            gray_img1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            error = mse(gray_img1, gray_img2)
            if error > 10.0:
                bad_frame_count += 1
            elif error > 1.0:
                refocus_count += 1

        remaining_frames = max(0, int(fps * duration) - frame_count)
        cv2.putText(frame, f"{width} x {height} @ {fps:.2f} Remaining: {remaining_frames}", org, font, fontScale, color, thickness)
        cv2.putText(frame, f"Bad Frames: {bad_frame_count}, Refocuses: {refocus_count}", (org[0], org[1] + 30), font, fontScale, color, thickness)
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
        "refocuses": refocus_count
    }

def main():
    """Main function to test multiple resolutions."""
    resolutions = [
        (640, 360),
        (640, 480),
        (800, 600),
        (1280, 720),
        (1920, 1080),
        (1600, 1200)
    ]
    total_tests = len(resolutions)
    successful_tests = 0
    total_frames_captured = 0
    results = []

    for width, height in resolutions:
        print(f"Starting test for resolution {width} x {height}")
        result = testWebcam(width, height, 10)
        if result:
            successful_tests += 1
            total_frames_captured += result["captured_frames"]
            results.append(f"Resolution {result['resolution']}: SUCCESS, Frames Captured: {result['captured_frames']}, Dropped: {result['dropped_frames']}, Bad: {result['bad_frames']}, Refocuses: {result['refocuses']}")
        else:
            results.append(f"Resolution {width} x {height}: FAILED")

    print("\nSummary of Results:")
    for result in results:
        print(result)
    print(f"\nTotal Resolutions Tested: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Failed Tests: {total_tests - successful_tests}")
    print(f"Total Frames Captured: {total_frames_captured}")

if __name__ == "__main__":
    main()
