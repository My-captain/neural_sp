# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-10-10 20:10
@file: data_prepare.py
"""
import glob
import json
import os
import shutil


def open_singer(base_path: str, target_dir: str):
    meta_info = []
    os.makedirs(target_dir, exist_ok=True)
    transcript = []
    target_wav = f"{target_dir}/wav"
    song_id = 0
    for sub_path in ["ManRaw", "WomanRaw"]:
        path = os.path.join(base_path, sub_path)
        songs = sorted(glob.glob(f"{path}/*"))
        for abs_path in songs:
            singer_id, song = abs_path.split("/")[-1].split("_")
            # singer_id = int(singer_id)
            song_id += 1
            os.makedirs(f"{target_wav}/S{song_id:06d}", exist_ok=True)
            for sample in glob.glob(f"{abs_path}/*.wav"):
                utt_id = sample.split("_")[-1].split(".")[0]
                sample_name = sample.split("/")[-1].split(".")[0]
                with open(f"{abs_path}/{sample_name}.txt", mode="r", encoding="utf8") as fp:
                    transcript_line = fp.readlines()[0]
                utt_id = int(utt_id)
                meta = {
                    "source": sample,
                    "id": f"openSingerS{song_id:06d}W{utt_id:04d}",
                }
                meta_info.append(meta)
                transcript.append(f"{meta['id']} {transcript_line}")
                shutil.copyfile(sample, f"{target_wav}/S{song_id:06d}/{meta['id']}.wav")
                pass

    with open(f"{target_dir}/transcript.txt", mode="a", encoding="utf8") as fp:
        fp.write("\n".join(transcript))
    json.dump(meta_info, open(f"{target_dir}/openSinger.json", mode="w", encoding="utf8"), ensure_ascii=False)


def pop_cs(base_path: str, target_dir: str):
    meta_info = []
    os.makedirs(target_dir, exist_ok=True)
    transcript = []
    target_wav = f"{target_dir}/wav"
    song_id = 1146
    songs = sorted(glob.glob(f"{base_path}/*"))
    for abs_path in songs:
        song_id += 1
        os.makedirs(f"{target_wav}/S{song_id:06d}", exist_ok=True)
        for sample in glob.glob(f"{abs_path}/*.wav"):
            utt_id = sample.split("/")[-1].split("_")[0]
            sample_name = sample.split("/")[-1].split(".")[0]
            with open(f"{abs_path}/{utt_id}.txt", mode="r", encoding="utf8") as fp:
                transcript_line = fp.readlines()[0].replace("@", " ")
            utt_id = int(utt_id)
            meta = {
                "source": sample,
                "id": f"popcsS{song_id:06d}W{utt_id:04d}",
            }
            meta_info.append(meta)
            transcript.append(f"{meta['id']} {transcript_line}")
            shutil.copyfile(sample, f"{target_wav}/S{song_id:06d}/{meta['id']}.wav")
            pass

    with open(f"{target_dir}/transcript.txt", mode="a", encoding="utf8") as fp:
        fp.write("\n".join(transcript))
    json.dump(meta_info, open(f"{target_dir}/popcs.json", mode="w", encoding="utf8"), ensure_ascii=False)


# open_singer("/media/zliu-elliot/Jarvis/OpenSinger", "/media/zliu-elliot/Jarvis/PopSong")
pop_cs("/media/zliu-elliot/Jarvis/popcs", "/media/zliu-elliot/Jarvis/PopSong")
