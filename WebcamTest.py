# Required pip install commands to run this script:
# pip install opencv-python
# pip install PyQt5
# pip install matplotlib
# pip install numpy
# pip install platformdirs
# pip install csv
# Ensure you have Python 3.6+ installed.
# For MacOS, ensure you have the latest version of PyQt5 and OpenCV installed.
# Run the script using: python WebcamTest.py

import sys
import cv2
import time
import platform
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class WebcamTest(QMainWindow):
    def __init__(self, requested_width, requested_height, duration=30):
        super().__init__()
        self.setWindowTitle(f"Webcam Test - {requested_width}x{requested_height}")
        self.requested_width = requested_width
        self.requested_height = requested_height
        self.duration = duration

        # Webcam setup
        if platform.system() == "Darwin":  # MacOS
            self.vc = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:  # Default to Windows (or other platforms)
            self.vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

        if not self.vc.isOpened():
            print("Error: Unable to open webcam.")
            sys.exit()

        self.width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vc.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            print("Error: Unable to determine FPS. Defaulting to 30.")
            self.fps = 30
        self.wait_time = int(1000.0 / self.fps)
        print(f"Actual resolution: {self.width} x {self.height}, FPS: {self.fps:.2f}")

        # Data tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.capture_times = []
        self.deltas = []
        self.avg_frame_rates = []
        self.prev_time = None

        # GUI setup
        self.init_ui()

        # Timer for updating webcam and charts
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam)
        self.timer.start(self.wait_time)

    def init_ui(self):
        """Initialize the UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Webcam display
        self.webcam_label = QLabel(self)
        self.webcam_label.setFixedSize(640, 480)

        # Matplotlib figure for charts
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax_deltas = self.fig.add_subplot(111)
        self.ax_deltas.set_title("Live Frame Capture Deltas and Average FPS")
        self.ax_deltas.set_xlabel("Time (seconds)")
        self.ax_deltas.set_ylabel("Delta Time (ms)", color="red")
        self.line_deltas, = self.ax_deltas.plot([], [], color="red", label="Delta Between Frames")
        self.ax_deltas.tick_params(axis="y", labelcolor="red")
        self.ax_deltas.grid(True)

        self.ax_fps = self.ax_deltas.twinx()
        self.ax_fps.set_ylabel("FPS", color="blue")
        self.line_avg_fps, = self.ax_fps.plot([], [], color="blue", label="Average Effective FPS")
        self.ax_fps.tick_params(axis="y", labelcolor="blue")

        self.canvas = FigureCanvas(self.fig)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.webcam_label)
        layout.addWidget(self.canvas)
        central_widget.setLayout(layout)

    def update_webcam(self):
        """Update the webcam frame and charts."""
        rval, frame = self.vc.read()
        if rval:
            self.frame_count += 1
            current_time = time.time() - self.start_time
            self.capture_times.append(current_time)

            if self.prev_time is not None:
                delta = current_time - self.prev_time
                self.deltas.append(delta)

                # Calculate average FPS using a rolling window with a max size of 24
                window_size = min(len(self.deltas), 24)
                avg_fps = 1 / (sum(self.deltas[-window_size:]) / window_size)
                self.avg_frame_rates.append(avg_fps)

            self.prev_time = current_time

            # Resize frame to 640x480 and convert to QImage
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.webcam_label.setPixmap(pixmap)

            # Update charts
            self.update_chart()

        if time.time() - self.start_time >= self.duration:
            print("Test completed.")
            self.timer.stop()
            self.vc.release()
            self.save_data_to_csv()

    def update_chart(self):
        """Update the live chart with new data."""
        if len(self.capture_times) > 1:
            # Update deltas
            deltas_x = self.capture_times[1:]
            deltas_y = [delta * 1000 for delta in self.deltas]  # Convert to milliseconds
            self.line_deltas.set_data(deltas_x, deltas_y)
            self.ax_deltas.set_xlim(0, max(deltas_x))
            self.ax_deltas.set_ylim(0, max(deltas_y) if deltas_y else 1)

        if len(self.avg_frame_rates) > 0:
            # Update average FPS
            fps_x = self.capture_times[-len(self.avg_frame_rates):]
            fps_y = self.avg_frame_rates
            self.line_avg_fps.set_data(fps_x, fps_y)
            self.ax_fps.set_xlim(0, max(fps_x))
            self.ax_fps.set_ylim(0, max(fps_y) if fps_y else 1)

        self.canvas.draw()

    def save_data_to_csv(self):
        """Save timestamps, deltas, and average effective frame rates to a CSV file."""
        print("Saving data to CSV...")
        with open("webcam_test_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp (s)", "Delta (ms)", "Average Effective FPS"])
            for i in range(len(self.capture_times)):
                timestamp = self.capture_times[i]
                delta = self.deltas[i] * 1000 if i < len(self.deltas) else ""
                avg_fps = self.avg_frame_rates[i] if i < len(self.avg_frame_rates) else ""
                writer.writerow([timestamp, delta, avg_fps])

def main():
    """Main function to test multiple resolutions with live charts."""
    app = QApplication(sys.argv)
    resolutions = [
        (640, 480),
        (1920, 1080)
    ]
    for width, height in resolutions:
        print(f"Starting test for resolution {width} x {height}")
        window = WebcamTest(width, height, 60)  # 60 seconds duration
        window.show()
        app.exec_()

if __name__ == "__main__":
    main()
