# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-10-10 20:10
@file: data_prepare.py
"""
import glob
import json
import os
import random
import shutil
import tqdm

import pandas as pd


def open_singer(base_path: str, target_dir: str, song_id=0):
    """
    
    :param base_path:
    :param target_dir:
    :param song_id: songID其实
    :return:
    """
    meta_info = []
    os.makedirs(target_dir, exist_ok=True)
    transcript = []
    target_wav = f"{target_dir}/wav/train"
    for sub_path in ["ManRaw", "WomanRaw"]:
        path = os.path.join(base_path, sub_path)
        songs = sorted(glob.glob(f"{path}/*"))
        for abs_path in tqdm.tqdm(songs):
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

    return song_id


def pop_cs(base_path: str, target_dir: str, song_id=0):
    meta_info = []
    os.makedirs(target_dir, exist_ok=True)
    transcript = []
    target_wav = f"{target_dir}/wav/train"
    # song_id = 1146
    songs = sorted(glob.glob(f"{base_path}/*"))
    for abs_path in tqdm.tqdm(songs):
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
    return song_id


def m4_singer(base_path: str, target_dir: str, song_id=0):
    meta_info = []
    os.makedirs(target_dir, exist_ok=True)
    transcript = []

    origin_meta = json.load(open(f"{base_path}/meta.json", mode="r", encoding="utf8"))
    meta_dict = dict()
    for item in origin_meta:
        meta_dict[item['item_name']] = {
            "phs": item['phs'],
            "notes_pitch": item['notes_pitch'],
            "notes_dur": item['notes_dur'],
            "ph_dur": item['ph_dur'],
            "txt": item['txt']
        }

    target_wav = f"{target_dir}/wav/train"
    songs = sorted(glob.glob(f"{base_path}/wav/*"))
    for abs_path in tqdm.tqdm(songs):
        song_id += 1
        os.makedirs(f"{target_wav}/S{song_id:06d}", exist_ok=True)
        for sample in sorted(glob.glob(f"{abs_path}/*.wav")):
            sample_name = sample.split("/")[-1].replace(".wav", "")
            transcript_line = meta_dict[sample_name]['txt']
            utt_id = sample_name.split("#")[-1]
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
    json.dump(meta_info, open(f"{target_dir}/m4_singer.json", mode="w", encoding="utf8"), ensure_ascii=False)


def opencpop(base_path: str, target_dir: str, song_id=0):
    # TODO: 待实现（需要切割音频成句子）
    meta_info = []
    os.makedirs(target_dir, exist_ok=True)
    transcript = []

    origin_meta = json.load(open(f"{base_path}/meta.json", mode="r", encoding="utf8"))
    meta_dict = dict()
    for item in origin_meta:
        meta_dict[item['item_name']] = {
            "phs": item['phs'],
            "notes_pitch": item['notes_pitch'],
            "notes_dur": item['notes_dur'],
            "ph_dur": item['ph_dur'],
            "txt": item['txt']
        }

    target_wav = f"{target_dir}/wav/train"
    songs = sorted(glob.glob(f"{base_path}/wav/*"))
    for abs_path in songs:
        song_id += 1
        os.makedirs(f"{target_wav}/S{song_id:06d}", exist_ok=True)
        for sample in sorted(glob.glob(f"{abs_path}/*.wav")):
            sample_name = sample.split("/")[-1].replace(".wav", "")
            transcript_line = meta_dict[sample_name]['txt']
            utt_id = sample_name.split("#")[-1]
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
    json.dump(meta_info, open(f"{target_dir}/m4_singer.json", mode="w", encoding="utf8"), ensure_ascii=False)


def split_test_from_train(train_dir: str, test_dir:str, test_sample:int):
    all_samples = list(glob.glob(f"{train_dir}/*/*.wav"))
    test_sample = random.choices(all_samples, k=test_sample)
    for wav in tqdm.tqdm(test_sample):
        [song_id, sample_name] = wav.split("/")[-2:]
        os.makedirs(f"{test_dir}/{song_id}", exist_ok=True)
        shutil.move(wav, f"{test_dir}/{song_id}/{sample_name}")
        pass


def modify_tsv(tsv_path: str, save_path: str):
    chunk = pd.read_csv(tsv_path, encoding='utf-8', delimiter='\t', chunksize=1000000)
    df = pd.concat(chunk)
    for index, row in df.iterrows():
        df.loc[index, 'feat_path'] = row['feat_path'].replace('/media/zliu-elliot/Jarvis/PopSong', '/home/zliu-elliot/dataset/PopSong')
    df.to_csv(save_path, encoding='utf-8', sep='\t', index=False)
    pass


# modify_tsv("/home/zliu-elliot/dataset/PopSong/dataset/dev_sp.tsv", "/home/zliu-elliot/dataset/PopSong/dataset/new_dev_sp.tsv")
# modify_tsv("/home/zliu-elliot/dataset/PopSong/dataset/train_sp.tsv", "/home/zliu-elliot/dataset/PopSong/dataset/new_train_sp.tsv")
# modify_tsv("/home/zliu-elliot/dataset/PopSong/dataset/test_sp.tsv", "/home/zliu-elliot/dataset/PopSong/dataset/new_test_sp.tsv")
song_id = 0
# song_id = open_singer("/home/zliu-elliot/dataset/OpenSinger", "/home/zliu-elliot/dataset/PopSong", song_id)
# song_id = pop_cs("/home/zliu-elliot/dataset/popcs", "/home/zliu-elliot/dataset/PopSong", song_id)
# song_id = m4_singer("/home/zliu-elliot/dataset/m4singer-short", "/home/zliu-elliot/dataset/PopSong", song_id)
# split_test_from_train("/home/zliu-elliot/dataset/PopSong/wav/train", "/home/zliu-elliot/dataset/PopSong/wav/test", 32)
