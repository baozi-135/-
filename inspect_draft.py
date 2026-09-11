import json, sys, os

for draft_id in ["12", "11", "10", "5", "1"]:
    path = f"D:/软件/剪映/JianyingPro Drafts/{draft_id}/draft_content.json"
    if not os.path.exists(path):
        print(f"=== Draft {draft_id}: NOT FOUND ===")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tracks = data.get('tracks', [])
    print(f"\n=== Draft {draft_id} ===")
    for t in tracks:
        segs = t.get('segments', [])
        ttype = t.get('type', '?')
        print(f"  {ttype}: {len(segs)} segments")
        if ttype == 'video' and segs:
            for i, s in enumerate(segs[:2]):
                tr = s.get('target_timerange', {})
                sr = s.get('source_timerange', {})
                tstart = tr.get('start', 0)
                tdur = tr.get('duration', 0)
                sstart = sr.get('start', 0) if sr else 0
                sdur = sr.get('duration', 0) if sr else 0
                print(f"    seg{i}: timeline ${tstart}-${tstart+tdur}us, source ${sstart}-${sstart+sdur}us")
    # Check materials
    mats = data.get('materials', {})
    videos = mats.get('videos', [])
    print(f"  materials.videos: {len(videos)}")
    for v in videos[:2]:
        vpath = v.get('local_path', v.get('path', 'N/A'))
        print(f"    - {os.path.basename(str(vpath))}")
