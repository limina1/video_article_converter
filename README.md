# Video Analysis and Content Generation Tool
You can see a produced video -> article transformation on our nostr client [Alexandria](https://next-alexandria.gitcitadel.eu/publication?d=nostr-apps-101)

An in depth description of the tool's usage on [habla.news](https://habla.news/u/liminal@gitcitadel.com/1729213258345)
This project is a Python-based tool for analyzing video content, extracting key frames, and generating summary content. It uses video entropy analysis to identify significant moments in the video and provides a graphical user interface for frame selection and content generation.

## Features

- Download YouTube videos with optional quality selection

- Analyze video entropy to identify significant moments

- Interactive GUI for visualizing video entropy and selecting key frames

- Automatic extraction of video chapters and subtitles

- Content generation based on selected frames and video context

- work only with transcripts for audio based videos

## Prerequisites

- Python 3.7+

- FFmpeg (for video processing)

## Installation

1. Clone this repository:

   ```

   git clone https://github.com/yourusername/video-analysis-tool.git

   cd video-analysis-tool

   ```

2. Set up a Conda environment (recommended):

   ```

   conda create -n video-analysis python=3.8

   conda activate video-analysis

   ```

3. Install the required packages:

   ```

   pip install -r requirements.txt

   ```

   Note: If you're not using Conda, you can still use pip to install the requirements in your preferred Python environment.

4. Install FFmpeg:

   - On macOS (using Homebrew): brew install ffmpeg

   - On Ubuntu/Debian: sudo apt-get install ffmpeg

   - On Windows: Download from [FFmpeg official site](https://ffmpeg.org/download.html) and add to PATH

## Usage

1. Run the main script:

   ```

   python video_analysis.py
   # or for a transcript
   python transcript_analysis.py

   ```


2. When prompted, enter the YouTube URL of the video you want to analyze.

3. Choose the desired video quality for download.

4. The tool will download the video, extract chapters and subtitles, and analyze the video entropy.

5. Use the interactive GUI to analyze the video:
   - The entropy plot will be displayed, showing entropy over time for each chapter.
   - You can make selections on the entropy plot to focus on specific parts of the video.
   - Enter the number of frames you want to select per chapter or selection (default is 5).
   Rule of thumb: High entropy, and also where the entropy signal is less stable (goes up and down) likely indicates there is more movement, or that the frame is not a diagram, a low and stable entropy signal likely means that there is a diagram being shown. You can validate this by opening up the video in your favorite video player.

6. The tool will generate content based on the selected frames and video context.

7. Output files (including generated content and selected frames) will be saved in the videos directory.

## Configuration

- API keys: Place your Anthropic API key in a file named video-articles.api in the /home/user/.apis/ directory.

## Troubleshooting

- If you encounter issues with video download, make sure you have the latest version of yt-dlp installed:

  ```

  pip install --upgrade yt-dlp

  ```

- For GUI-related issues, ensure you have the required Qt libraries installed:

  ```

  conda install pyqt

  ```

  or

  ```

  pip install PyQt5

  ```

In the case of an error with the following message:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, minimal, minimalegl, offscreen, vnc, xcb.

Aborted (core dumped)
```
You can fix it by deleting the following file:

``` sh
/home/user/.conda/envs/video-analysis/lib/python3.10/site-packages/PyQt5/Qt/plugins/platforms/libqxcb.so
# or similarly for pip
/home/user/.local/lib/python3.10/site-packages/PyQt5/Qt/plugins/platforms/libqxcb.so
```
See [this issue](https://stackoverflow.com/questions/59809703/could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-found) for more information.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
