from moviepy.editor import ImageSequenceClip
import glob


image_dir = glob.glob('./temp/*.png')

# Create a video clip from the PNG files
image_clip = ImageSequenceClip(image_dir, fps=15)

# Write the final video file
image_clip.write_videofile('output_video.mp4')
