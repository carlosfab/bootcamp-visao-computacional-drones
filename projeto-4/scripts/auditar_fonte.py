#!/usr/bin/env python3
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from remote_inventory import RangeReader, OUT

reader = RangeReader()
metadata = {}
with zipfile.ZipFile(reader) as archive:
    for item in archive.infolist():
        if not item.is_dir() and item.filename.lower().endswith((".txt", ".xml", ".names", ".data")):
            raw = archive.read(item)
            try:
                metadata[item.filename] = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                metadata[item.filename] = raw.decode("cp1252")
(OUT / "icaerus_annotations.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
inventory = json.loads((OUT / "icaerus_inventory.json").read_text())
index = defaultdict(list)
for path, content in metadata.items():
    p = Path(path)
    if p.suffix == ".txt" and p.name != "train.txt":
        index[(p.parts[1], p.stem)].append((path, content))

rows = []
anomalies = []
class_names = {path: data for path, data in metadata.items() if path.endswith(".names")}
groups = defaultdict(lambda: Counter(images=0, positive=0, empty=0, missing=0, boxes=0))
for image in inventory:
    p = Path(image["path"])
    if p.suffix.lower() not in (".jpg", ".jpeg"):
        continue
    farm, flight = p.parts[1], p.parts[3]
    candidates = index.get((farm, p.stem), [])
    stat = groups[(farm, flight)]
    stat["images"] += 1
    if len(candidates) != 1:
        anomalies.append({"image":image["path"], "label_candidates":[c[0] for c in candidates]})
        stat["missing"] += 1
        continue
    label_path, content = candidates[0]
    labels = [line.split() for line in content.splitlines() if line.strip()]
    for label in labels:
        if len(label) != 5:
            raise ValueError((label_path, label))
        coords = list(map(float, label[1:]))
        x, y, w, h = coords
        if not all(0 <= c <= 1 for c in coords) or w <= 0 or h <= 0 or min(x-w/2,y-h/2) < -1e-5 or max(x+w/2,y+h/2) > 1+1e-5:
            anomalies.append({"label":label_path,"out_of_range":label})
    classes = Counter(label[0] for label in labels)
    stat["boxes"] += len(labels)
    stat["positive" if labels else "empty"] += 1
    rows.append({**image, "farm":farm,"flight":flight,"date":flight.split('_')[1][:8],
                 "label_path":label_path,"n_boxes":len(labels),"class_ids":dict(classes)})
(OUT / "icaerus_image_metadata.json").write_text(json.dumps(rows, indent=2))
report = {"groups":[{"farm":f,"flight":v,**stat} for (f,v),stat in groups.items()],
          "class_names":class_names,"anomalies":anomalies,
          "requests":reader.requests,"bytes_transferred":reader.transferred}
(OUT / "icaerus_audit.json").write_text(json.dumps(report, indent=2))
voc = defaultdict(list)
for path, content in metadata.items():
    if path.endswith('.xml'):
        voc[(Path(path).parts[1], Path(path).stem)].append((path, ET.fromstring(content).findall('object')))
differences = []
paired_voc = 0
for row in rows:
    matches = voc.get((row['farm'],Path(row['path']).stem), [])
    if len(matches) != 1:
        continue
    paired_voc += 1
    path, objects = matches[0]
    if len(objects) != row['n_boxes']:
        differences.append({'image':row['path'],'yolo_count':row['n_boxes'],
                            'voc_count':len(objects),
                            'voc_names':dict(Counter(obj.findtext('name') for obj in objects))})
summary = {'schema_version':1,'source_doi':'10.5281/zenodo.11048412',
           'source_images':sum(r['path'].lower().endswith(('.jpg','.jpeg')) for r in inventory),
           'matched_yolo_labels':len(rows), 'source_flights':len(groups),
           'positive_images':sum(r['n_boxes']>0 for r in rows),
           'present_empty_labels':sum(r['n_boxes']==0 for r in rows),
           'source_yolo_boxes':sum(r['n_boxes'] for r in rows),
           'class_box_counts':dict(sum((Counter(r['class_ids']) for r in rows),Counter())),
           'source_advertised_boxes':4941,'annotation_anomalies':anomalies,
           'uniquely_paired_voc_images':paired_voc,'voc_yolo_count_differences':differences,
           'scope':'Structural inventory and labels. Does not establish visual completeness of annotations.'}
assets = Path(__file__).resolve().parents[1]/'assets'
assets.mkdir(exist_ok=True)
(assets/'auditoria-fonte.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({"images":len(rows), "boxes":sum(r["n_boxes"] for r in rows),
                  "anomalies":len(anomalies), "report":str(OUT / "icaerus_audit.json")}, indent=2))
