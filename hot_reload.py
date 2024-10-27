from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time
import os

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, command):
        self.command = command
        self.process = None
        self.start_process()

    def start_process(self):
        if self.process:
            self.process.terminate()
        self.process = subprocess.Popen(self.command, shell=True)

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"{event.src_path} has been modified, restarting...")
            self.start_process()

if __name__ == "__main__":
    command = "python app.py"  # Replace with the actual script name
    path = os.path.dirname(os.path.abspath(__file__))

    event_handler = ChangeHandler(command)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
