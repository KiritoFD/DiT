import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(path):
    return str(path).strip().replace('\\', '/')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        result = list(csv.DictReader(f))
    if not result:
        raise ValueError(f'Empty CSV: {path}')
    for field in ('image_path', 'calligrapher', 'script', 'character'):
        if field not in result[0]:
            raise ValueError(f'Missing {field}: {path}')
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, items, fields=None):
    items = list(items)
    fields = fields or list(dict.fromkeys(k for row in items for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


class Groups:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a,b)] = min(a,b)

    def labels(self):
        return [self.find(i) for i in range(len(self.parent))]
