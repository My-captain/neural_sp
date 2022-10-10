# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-10-10 20:10
@file: data_prepare.py
"""
import glob
import os


def open_singer(base_path: str):
    for sub_path in ["ManRaw", "WomanRaw"]:
        path = os.path.join(base_path, sub_path)
        songs = glob.glob(path)


open_singer("/media/Jarvis/OpenSinger/")

