import sys
import os


def ensemble_path(root, models):
  paths = [os.path.join(root, model) for model in models]
  # for path in paths:
  #   if not os.path.exists(path):
  #     raise ValueError(path)
  return ":".join(paths)


if __name__ == "__main__":
  root = sys.argv[1]
  models = sys.argv[2:]
  print(ensemble_path(root, models))
