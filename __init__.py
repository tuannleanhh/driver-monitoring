from main import run_on_video

if __name__ == '__main__':
    idx = 3
    video_path = f'testing_media/testing_videos/video_{idx}.mp4'
    save_dir = f'outputs/video_{idx}_output.avi'

    run_on_video(video_path, save_dir, plot=True, save=True)
