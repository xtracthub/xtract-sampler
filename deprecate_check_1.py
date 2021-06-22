
import os
import subprocess

os.chdir('/src/')

model = "rf"
features = "randhead"

runnable = """python3 FTI_Models/main.py /src/ {0} {1}""".format(str(model), str(features))

processes = [subprocess.Popen(runnable, shell=True)]
for p in processes: p.wait()
