import os
import moviepy.editor as mp

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip


all_videos = r'/vggsound.csv'
all_audio_dir = r'/workdir/carrot/VGGSound/Audios'
if not os.path.exists(all_audio_dir):
    os.makedirs(all_audio_dir)

# train set processing
with open(all_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    print(item)
    item = item.split(',')
    video_name = item[0] + '_' + str(item[1]).zfill(6)
    mp4_filename = os.path.join(r'/workdir/carrot/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video', video_name + '.mp4')
    wav_filename = os.path.join(all_audio_dir, video_name+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        my_clip = extract_audio(mp4_filename)
        my_clip.audio.write_audiofile(wav_filename)

        #os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

