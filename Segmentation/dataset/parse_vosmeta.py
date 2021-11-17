import json
import os


ROOT_DIR = "/orion/u/yzcong/datasets/train"

with open(f'{ROOT_DIR}/meta.json') as f:
    meta = json.load(f)
videos = meta['videos']
# print('Num of videos', len(meta))
# print(meta[list(meta.keys())[0]].keys())

# videos:
# -- 172t171ey1y:
# ---- objects:
# ------ '1':
# -------- category: str
# -------- frames: list of strs
# ------ '2':
# ------ ...
# -- ...

def extract_meta(video):
    file_path = os.path.join(ROOT_DIR, f'Annotations/{video}/meta.json')
    data = videos[video]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

all_cats = set()
cat_instance = dict()
for video in videos:
    for k, v in videos[video]['objects'].items():
        all_cats.add(v['category'])
        if v['category'] not in cat_instance:
            cat_instance[v['category']] = []
        flag = False
        for instance in cat_instance[v['category']]:
            if instance['video'] == video:
                flag = True
                instance['id'].append(k)
        if not flag:
            cat_instance[v['category']].append({'video': video, 'id': [k], 'frames': len(v['frames'])})
print(all_cats)
for k, v in cat_instance.items():
    print(k, sum([1 for x in v if len(x['id']) == 1]), max([x['frames'] for x in v if len(x['id']) == 1]))

# file_path = os.path.join(ROOT_DIR, 'cat2video.json')
# with open(file_path, 'w', encoding='utf-8') as f:
#     json.dump(cat_instance, f, ensure_ascii=False, indent=4)

# for video in videos:
#     extract_meta(video)

