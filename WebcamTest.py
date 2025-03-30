import cv2
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv

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
    avg_frame_rates = []  # Track average effective frame rates
    prev_time = None  # Initialize prev_time
    timeout_start = time.time()  # Initialize timeout_start

    # Tkinter setup
    root = tk.Tk()
    root.title(f"Webcam Test - {requested_width}x{requested_height}")

    # Webcam frame on the left
    webcam_label = ttk.Label(root)
    webcam_label.grid(row=0, column=0, padx=10, pady=10)

    # Chart for deltas on the right
    fig_deltas = Figure(figsize=(5, 4), dpi=100)
    ax_deltas = fig_deltas.add_subplot(111)
    ax_deltas.set_title("Live Frame Capture Deltas")
    ax_deltas.set_xlabel("Time (seconds)")
    ax_deltas.set_ylabel("Delta Time (ms)")
    line_deltas, = ax_deltas.plot([], [], color="red", label="Delta Between Frames")
    ax_deltas.legend()
    ax_deltas.grid(True)

    canvas_deltas = FigureCanvasTkAgg(fig_deltas, master=root)
    canvas_deltas_widget = canvas_deltas.get_tk_widget()
    canvas_deltas_widget.grid(row=0, column=1, padx=10, pady=10)

    # Chart for average FPS on the far right
    fig_fps = Figure(figsize=(5, 4), dpi=100)
    ax_fps = fig_fps.add_subplot(111)
    ax_fps.set_title("Live Average Effective FPS")
    ax_fps.set_xlabel("Time (seconds)")
    ax_fps.set_ylabel("FPS")
    line_avg_fps, = ax_fps.plot([], [], color="blue", label="Average Effective FPS")
    ax_fps.legend()
    ax_fps.grid(True)

    canvas_fps = FigureCanvasTkAgg(fig_fps, master=root)
    canvas_fps_widget = canvas_fps.get_tk_widget()
    canvas_fps_widget.grid(row=0, column=2, padx=10, pady=10)

    running = True

    def update_webcam():
        """Update the webcam frame."""
        nonlocal frame_count, capture_times, deltas, avg_frame_rates, prev_time, timeout_start
        timeout_duration = 5  # Timeout duration in seconds

        rval, frame = vc.read()
        if rval:
            timeout_start = time.time()  # Reset timeout on successful frame capture
            frame_count += 1
            current_time = time.time() - start_time
            capture_times.append(current_time)

            if prev_time is not None:
                delta = current_time - prev_time
                deltas.append(delta)

                # Calculate average FPS using a rolling window with a max size of 24
                window_size = min(len(deltas), 24)
                avg_fps = 1 / (sum(deltas[-window_size:]) / window_size)
                avg_frame_rates.append(avg_fps)

            prev_time = current_time

            # Resize frame to 640x480 before displaying
            frame = cv2.resize(frame, (640, 480))
            # Convert frame to RGB and display in Tkinter
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes())
            webcam_label.image = frame_image  # Persist the PhotoImage object
            webcam_label.configure(image=frame_image)

        if time.time() - start_time >= duration or not running:
            print("Test completed or window closed.")
            vc.release()
            root.quit()
        else:
            root.after(wait_time, update_webcam)

    def update_chart():
        """Update the live charts with new data."""
        # Update deltas chart
        if len(capture_times) > 1:
            deltas_x = capture_times[1:]
            deltas_y = [delta * 1000 for delta in deltas]  # Convert to milliseconds
            line_deltas.set_data(deltas_x, deltas_y)
            ax_deltas.set_xlim(0, max(deltas_x))
            ax_deltas.set_ylim(0, max(deltas_y) if deltas_y else 1)
            canvas_deltas.draw()

        # Update average FPS chart
        if len(avg_frame_rates) > 0:
            fps_x = capture_times[-len(avg_frame_rates):]
            fps_y = avg_frame_rates
            line_avg_fps.set_data(fps_x, fps_y)
            ax_fps.set_xlim(0, max(fps_x))
            ax_fps.set_ylim(0, max(fps_y) if fps_y else 1)
            canvas_fps.draw()

        if running:
            root.after(500, update_chart)

    def save_data_to_csv():
        """Save timestamps, deltas, and average effective frame rates to a CSV file."""
        print("Saving data to CSV...")
        with open("webcam_test_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            print("Writing header to CSV...")
            writer.writerow(["Timestamp (s)", "Delta (ms)", "Average Effective FPS"])
            for i in range(len(capture_times)):
                timestamp = capture_times[i]
                delta = deltas[i] * 1000 if i < len(deltas) else ""
                avg_fps = avg_frame_rates[i - 24] if i >= 24 else ""
                writer.writerow([timestamp, delta, avg_fps])

    def on_close():
        """Handle the window close event."""
        nonlocal running
        running = False
        vc.release()
        save_data_to_csv()  # Save data to CSV before closing
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
        "deltas": deltas,
        "avg_frame_rates": avg_frame_rates
    }

def main():
    """Main function to test multiple resolutions with live charts."""
    resolutions = [
        #(640, 480),
        (1920, 1080)
    ]
    results = []

    for width, height in resolutions:
        print(f"Starting test for resolution {width} x {height}")
        result = testWebcamLiveChart(width, height, 60)  # 60 seconds duration
        if result:
            results.append(result)
            print(f"Resolution {result['resolution']}: SUCCESS, Frames Captured: {result['captured_frames']}")
        else:
            print(f"Resolution {width} x {height}: FAILED")

    print("\nSummary of Results:")
    for result in results:
        print(f"Resolution {result['resolution']}: Captured Frames: {result['captured_frames']}")

    # Save all results to a CSV file
    print("Saving all results to CSV...")
    with open("webcam_test_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Resolution", "Timestamp (s)", "Delta (ms)", "Average Effective FPS"])
        for result in results:
            resolution = result["resolution"]
            capture_times = result["deltas"]
            deltas = result["deltas"]
            avg_frame_rates = result["avg_frame_rates"]
            for i in range(len(capture_times)):
                timestamp = capture_times[i]
                delta = deltas[i] * 1000 if i < len(deltas) else ""
                avg_fps = avg_frame_rates[i - 24] if i >= 24 else ""
                writer.writerow([resolution, timestamp, delta, avg_fps])

if __name__ == "__main__":
    main()
