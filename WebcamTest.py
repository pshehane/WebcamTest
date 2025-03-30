import cv2
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
    prev_time = None  # Initialize prev_time
    timeout_start = time.time()  # Initialize timeout_start

    # Tkinter setup
    root = tk.Tk()
    root.title(f"Webcam Test - {requested_width}x{requested_height}")

    # Webcam frame on the left
    webcam_label = ttk.Label(root)
    webcam_label.grid(row=0, column=0, padx=10, pady=10)

    # Chart on the right
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Live Frame Capture Deltas")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Delta Time (seconds)")
    line, = ax.plot([], [], color="red", label="Delta Between Frames")
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=1, padx=10, pady=10)

    running = True

    def update_chart():
        """Update the live chart with new data."""
        if len(capture_times) > 1:
            line.set_data(capture_times[1:], deltas)
            ax.set_xlim(0, max(capture_times))
            ax.set_ylim(0, max(deltas) if deltas else 1)
            canvas.draw()

        if running:
            root.after(500, update_chart)

    def update_webcam():
        """Update the webcam frame."""
        nonlocal frame_count, capture_times, deltas, prev_time, timeout_start
        timeout_duration = 5  # Timeout duration in seconds

        rval, frame = vc.read()
        if rval:
            timeout_start = time.time()  # Reset timeout on successful frame capture
            frame_count += 1
            current_time = time.time() - start_time
            capture_times.append(current_time)

            if prev_time is not None:
                deltas.append(current_time - prev_time)

            prev_time = current_time

            # Convert frame to RGB and display in Tkinter
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes())
            webcam_label.image = frame_image  # Persist the PhotoImage object
            #print(f"Frame {frame_count} captured at {current_time:.2f} seconds")
            webcam_label.configure(image=frame_image)

        if time.time() - start_time >= duration or not running:
            print("Test completed or window closed.")
            vc.release()
            root.quit()
        else:
            root.after(wait_time, update_webcam)

    def on_close():
        """Handle the window close event."""
        nonlocal running
        running = False
        vc.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_chart()
    update_webcam()
    root.mainloop()

    # Ensure the root window is destroyed after the test
    if root.winfo_exists():
        root.destroy()

    return {
        "resolution": f"{width} x {height}",
        "captured_frames": frame_count,
        "deltas": deltas
    }

def main():
    """Main function to test multiple resolutions with live charts."""
    resolutions = [
        (640, 480),
        (1920, 1080)
    ]
    results = []

    for width, height in resolutions:
        print(f"Starting test for resolution {width} x {height}")
        result = testWebcamLiveChart(width, height, 5)
        if result:
            results.append(result)
            print(f"Resolution {result['resolution']}: SUCCESS, Frames Captured: {result['captured_frames']}")
        else:
            print(f"Resolution {width} x {height}: FAILED")

    print("\nSummary of Results:")
    for result in results:
        print(f"Resolution {result['resolution']}: Captured Frames: {result['captured_frames']}")

if __name__ == "__main__":
    main()
