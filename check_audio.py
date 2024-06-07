import torch
import torchaudio


def load_audio(audiopath, sampling_rate):
    audio, sr = torchaudio.load(audiopath)
    #print(f"wave shape: {audio.shape}, sample_rate: {sr}")

    if audio.size(0) > 1:  # mix to mono
        audio = audio[0].unsqueeze(0)

    if sr != sampling_rate:
        try:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        except Exception as e:
            print(f"Warning: {audiopath}, wave shape: {audio.shape}, sample_rate: {sr}")
            return None
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


filelist = "filelist/val.list"


def main():
    with open(filelist, 'r', encoding='utf8') as fin:
        for line in fin:
            full_path = line.strip()
            try:
                wav = load_audio(full_path, 24000)
                if wav is None:
                    print(f"Warning: {full_path} loading error, skip!")
            except:
                print(f"Warning: {full_path} processing error, skip!")


if __name__ == '__main__':
    main()

