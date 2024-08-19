import glob
import os

def load_file(dir: str, filename: str) -> str:
  return open(os.path.join(dir, filename), 'r').read()


def load_files(dir: str, pattern: str) -> list[str]:
  doc_paths = sorted(glob.glob(pattern, root_dir=dir))
  return [load_file(dir, doc_name) for doc_name in doc_paths]
