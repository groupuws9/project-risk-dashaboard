import os
root = r"d:\new research topics\project risk dashaboard\ml_pipeline"
for f in os.listdir(root):
    print(os.path.join(root, f))