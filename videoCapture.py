from js import document

class VideoCapture:

    def __init__(self):
        pass

    def startCapture(self, event):
        video = document.getElementById('recordingPreview').src
        print(video)
        

videoCapture = VideoCapture()