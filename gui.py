import sys
import numpy as np
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import cv2
import json
import os
from datetime import datetime
def load_video_data(filename='processed_video_data.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['video_data'], data['fps'], data['chapters']

class EntropyPlotWidget(QWidget):
    def __init__(self, video_data, fps, chapters):
        super().__init__()
        self.video_data = video_data
        self.fps = fps
        self.chapters = chapters
        self.selections = []
        self.current_selection = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Add NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.plot_entropy()

        # Set up RectangleSelector for zooming
        self.zoom_selector = RectangleSelector(
            self.ax, self.zoom_function,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # Connect mouse events for selection
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def plot_entropy(self):
        self.ax.clear()
        for chapter in self.video_data:
            self.ax.plot(chapter['timestamps'], chapter['entropies'], label=chapter['title'])

        # Add chapter boundaries
        ylim = self.ax.get_ylim()
        for i, chapter in enumerate(self.chapters):
            start_time = chapter['start_time']
            if i < len(self.chapters) - 1:
                end_time = self.chapters[i+1]['start_time']
            else:
                end_time = max(c['timestamps'][-1] for c in self.video_data)

            self.ax.axvline(x=start_time, color='r', linestyle='--', alpha=0.5)
            self.ax.text(start_time, ylim[1], f" {chapter['title']}", rotation=90, verticalalignment='top')

        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Entropy')
        self.ax.set_title('Video Entropy Over Time')
        self.ax.legend()
        self.canvas.draw()

    def zoom_function(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.ax.set_xlim(min(x1, x2), max(x1, x2))
        self.ax.set_ylim(min(y1, y2), max(y1, y2))
        self.canvas.draw()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax and self.toolbar.mode == '':
            self.current_selection = event.xdata

    def on_mouse_release(self, event):
        if event.inaxes == self.ax and self.current_selection is not None and self.toolbar.mode == '':
            start = min(self.current_selection, event.xdata)
            end = max(self.current_selection, event.xdata)
            self.selections.append((start, end))
            self.draw_selections()
            self.current_selection = None

    def on_mouse_move(self, event):
        if event.inaxes == self.ax and self.current_selection is not None and self.toolbar.mode == '':
            self.draw_selections(temp_selection=(self.current_selection, event.xdata))

    def draw_selections(self, temp_selection=None):
        self.ax.clear()
        self.plot_entropy()
        ylim = self.ax.get_ylim()
        for start, end in self.selections:
            rect = Rectangle((start, ylim[0]), end - start, ylim[1] - ylim[0],
                             facecolor='yellow', alpha=0.3)
            self.ax.add_patch(rect)
        if temp_selection:
            start, end = min(temp_selection), max(temp_selection)
            rect = Rectangle((start, ylim[0]), end - start, ylim[1] - ylim[0],
                             facecolor='red', alpha=0.3)
            self.ax.add_patch(rect)
        self.canvas.draw()
class MainWindow(QMainWindow):
    def __init__(self, video_data, fps, chapters, video_folder):
        super().__init__()
        self.video_data = video_data
        self.fps = fps
        self.chapters = chapters
        self.video_folder = video_folder
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Video Entropy Analysis')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.entropy_plot = EntropyPlotWidget(self.video_data, self.fps, self.chapters)
        layout.addWidget(self.entropy_plot)

        select_button = QPushButton('Select Frames')
        select_button.clicked.connect(self.on_select_frames)
        layout.addWidget(select_button)

        self.setCentralWidget(central_widget)
    # def on_select_frames(self):
        selections = self.entropy_plot.selections
        if not selections:
            print("No selections made. Please select ranges in the entropy plot first.")
            return
        frame_selection_window = FrameSelectionWindow(self.video_data, selections, self.fps, self)
        if frame_selection_window.exec_() == QDialog.Accepted:
            selected_frames = frame_selection_window.get_selected_frames()
            self.save_selected_frames(selected_frames)#

    # def on_select_frames(self):
        selections = self.entropy_plot.selections
        if not selections:
            print("No selections made. Please select ranges in the entropy plot first.")
            return
        frame_selection_window = FrameSelectionWindow(self.video_data, selections, self.fps, self, frame_interval=5)
        if frame_selection_window.exec_() == QDialog.Accepted:
            selected_frames = frame_selection_window.get_selected_frames()
            self.save_selected_frames(selected_frames)#
    def on_select_frames(self):
        selections = self.entropy_plot.selections
        if not selections:
            print("No selections made. Please select ranges in the entropy plot first.")
            return
        frame_selection_window = FrameSelectionWindow(self.video_data, selections, self.fps, self, num_frames=5)
        if frame_selection_window.exec_() == QDialog.Accepted:
            selected_frames = frame_selection_window.get_selected_frames()
            self.save_selected_frames(selected_frames)
    def save_selected_frames(self, selected_frames):
        # Use the video folder path from the video download process
        video_folder = self.video_folder  # Assume this is set during video download

        # Create a subdirectory for selected frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(video_folder, f"selected_frames_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Prepare metadata
        metadata = []

        for identifier, pixmap in selected_frames.items():
            selection_index, frame_index = map(int, identifier.split(','))

            # Find the correct chapter and frame
            frame_found = False
            for chapter in self.video_data:
                if frame_index < len(chapter['frames']):
                    frame = chapter['frames'][frame_index]
                    frame_timestamp = chapter['timestamps'][frame_index]
                    frame_found = True
                    break
                frame_index -= len(chapter['frames'])

            if not frame_found:
                print(f"Warning: Could not find frame for identifier {identifier}")
                continue

            # Save the frame as an image
            frame_filename = f"frame_{selection_index}_{frame_index}.png"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Add metadata
            metadata.append({
                "chapter_title": chapter['title'],
                "timestamp": frame_timestamp,
                "filename": frame_filename
            })

        # Save metadata
        metadata_file = os.path.join(save_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Selected frames and metadata saved in {save_dir}")

        # Update the interactive entropy plot (if needed)
        self.update_entropy_plot_with_selections(metadata)

    def update_entropy_plot_with_selections(self, metadata):
        # Clear previous selections
        for artist in self.entropy_plot.ax.get_children():
            if isinstance(artist, Rectangle) and artist.get_facecolor() == (1, 1, 0, 0.3):  # Yellow with alpha
                artist.remove()

        # Add new selections
        for item in metadata:
            timestamp = item['timestamp']
            self.entropy_plot.ax.axvline(x=timestamp, color='g', linestyle='--', alpha=0.7)

        self.entropy_plot.canvas.draw()

class FrameSelectionWindow(QDialog):
    def __init__(self, video_data, selections, fps, parent=None, num_frames=5):
        super().__init__(parent)
        self.video_data = video_data
        self.selections = selections
        self.fps = fps
        self.num_frames = num_frames
        self.selected_frames = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Frame Selection')
        self.setGeometry(200, 200, 800, 600)

        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        for i, (start, end) in enumerate(self.selections):
            row_label = QLabel(f"Selection {i+1}: {start:.2f}s - {end:.2f}s")
            scroll_layout.addWidget(row_label, i, 0)

            frames = self.get_frames_for_selection(start, end)
            for j, (frame, identifier) in enumerate(frames):
                frame_label = ClickableLabel(frame, identifier)
                frame_label.clicked.connect(self.toggle_frame)
                scroll_layout.addWidget(frame_label, i, j+1)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        save_button = QPushButton("Save Selected Frames")
        save_button.clicked.connect(self.save_frames)
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)

    def save_frames(self):
        self.accept()

    def toggle_frame(self, identifier):
        sender = self.sender()
        if identifier in self.selected_frames:
            del self.selected_frames[identifier]
            sender.setStyleSheet("")
        else:
            self.selected_frames[identifier] = sender.pixmap()
            sender.setStyleSheet("border: 3px solid red;")

    def get_selected_frames(self):
        return self.selected_frames

    def get_frames_for_selection(self, start, end):
        frames = []
        selection_frames = []

        for chapter in self.video_data:
            chapter_frames = chapter['frames']
            chapter_timestamps = chapter['timestamps']

            for frame, timestamp in zip(chapter_frames, chapter_timestamps):
                if start <= timestamp <= end:
                    selection_frames.append((frame, timestamp))

        # Divide the selection into num_frames parts and pick the first frame from each part
        total_frames = len(selection_frames)
        if total_frames == 0:
            return frames

        indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]

        for i, idx in enumerate(indices):
            frame, timestamp = selection_frames[idx]
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            frames.append((pixmap, f"{i},{idx}"))

        return frames

class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)
    def __init__(self, pixmap, identifier, timestamp=None):
        super().__init__()
        self.setPixmap(pixmap)
        self.identifier = identifier
        self.timestamp = timestamp
    def mousePressEvent(self, event):
        self.clicked.emit(self.identifier)
# app = QApplication(sys.argv)
# print("Loading video data...")
# video_data, fps, chapters = load_video_data('test_data.pkl')
# print("Video data loaded.")
# main_window = MainWindow(video_data, fps, chapters)
# main_window.show()
def main(video_data, fps, chapters, video_folder):
    app = QApplication(sys.argv)
    print("Loading video data...")
    print("Video data loaded.")
    main_window = MainWindow(video_data, fps, chapters, video_folder)
    main_window.show()
    app.exec_()
    del app
