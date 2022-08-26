import subprocess, os
import numpy as np
from analysis.utils import createFile, deleteFile

def cvnn_formula(samples, labels, labelsTheta):
    # Defining the R script and loading the instance in Python
    createFile(samples, labels, cvnn = True, labels2 = labelsTheta)
    score = command()
    # deleteFile()        

    return score

def command():
    command = 'Rscript'
    # command = 'Rscript'                    # OR WITH bin FOLDER IN PATH ENV VAR 
    arg = '--vanilla' 

    try: 
        p = subprocess.Popen([command, arg,
                            "analysis/cqcluster/cvnn.R"],
                            cwd = os.getcwd(),
                            stdin = subprocess.PIPE, 
                            stdout = subprocess.PIPE, 
                            stderr = subprocess.PIPE) 

        output, error = p.communicate() 

        if p.returncode == 0: 
            # print('R OUTPUT:\n {0}'.format(output.decode("utf-8"))) 
            out = output.decode("utf-8")
            lines = out.split('\n')
            line = lines[1].replace('[1]', '')
            res = line.split()
            return (float(res[0]), float(res[1]))
        else: 
            print('R ERROR:\n {0}'.format(error.decode("utf-8"))) 
            return None

    except Exception as e: 
        print("dbc2csv - Error converting file: ") 
        print(e)

        return False