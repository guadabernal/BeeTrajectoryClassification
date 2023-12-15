#pip install ffmpeg-python
import ffmpeg 
import os

def h264_to_avi(input_file, output_file, num_threads):
    try:
        input_stream = ffmpeg.input(input_file)
        output_stream = ffmpeg.output(input_stream, output_file, vcodec='mpeg4', threads=num_threads)
        ffmpeg.run(output_stream, overwrite_output=True)
        print(f"Conversion complete for {output_file}.")
    except ffmpeg.Error as e:
        print("An error occurred:", e.stderr)

def convert_directory(source_folder, destination_folder, num_threads):
    for file in os.listdir(source_folder):
        if file.endswith(".h264"):
            input_file = os.path.join(source_folder, file)
            output_file = os.path.join(destination_folder, file.replace('.h264', '.avi'))

            if not os.path.exists(output_file):
                h264_to_avi(input_file, output_file, num_threads)
            else:
                print(f"File {output_file} already exists. Skipping conversion.")

if __name__ == "__main__":
    source_folder = 'videos_raw'
    destination_folder = 'videos_avi'
    num_processors = 8  # set number of processors

    convert_directory(source_folder, destination_folder, num_processors)
