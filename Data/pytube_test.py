import pytube
path = "C:/Users/Abhinav/vf_data" # substitute to where you want to place the downloaded file

def YTDownload(link):
    youTubeObject = pytube.YouTube(link)
    youTubeObject = youTubeObject.streams.get_highest_resolution()
    
    try:
        if youTubeObject is not None:
            youTubeObject.download(path)
    except:
        print("There has been an error.")
    
    print("All good.")
    
link = ("youtube.com/watch?v=hDkLoRyiIeI")
YTDownload(link)
