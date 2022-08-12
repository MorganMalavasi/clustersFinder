import numpy as np
import subprocess, os
from analysis.utils import createFile, deleteFile
from scipy import stats


def pearson_index(data, labels, matrixOfDissimilarities):

    # vectorization of the matrix
    size = ((matrixOfDissimilarities.shape[0] * matrixOfDissimilarities.shape[1]) // 2) - (matrixOfDissimilarities.shape[0] // 2)
    vectorDissimilarities = np.zeros(size)
    vectorOf1 = np.zeros(size)
    counter = 0
    for i in range (matrixOfDissimilarities.shape[0]):
        for j in range(i + 1, matrixOfDissimilarities.shape[1]):
            vectorDissimilarities[counter] = matrixOfDissimilarities[i, j]
            if labels[i] != labels[j]:
                vectorOf1[counter] = counter
            else:
                vectorOf1[counter] = 0
            counter += 1

    res = stats.pearsonr(vectorDissimilarities, vectorOf1)
    return res[1]

def pearson_index_R(samples, labels):
    # Defining the R script and loading the instance in Python
    createFile(samples, labels)
    score = command()
    deleteFile()        

    return score

def command():
    command = 'Rscript'
    # command = 'Rscript'                    # OR WITH bin FOLDER IN PATH ENV VAR 
    arg = '--vanilla' 

    try: 
        p = subprocess.Popen([command, arg,
                            "analysis/cqcluster/pearson.R"],
                            cwd = os.getcwd(),
                            stdin = subprocess.PIPE, 
                            stdout = subprocess.PIPE, 
                            stderr = subprocess.PIPE) 

        output, error = p.communicate() 

        if p.returncode == 0: 
            # print('R OUTPUT:\n {0}'.format(output.decode("utf-8"))) 
            out = output.decode("utf-8")
            out = out.replace('[1]', '')
            return float(out)
        else: 
            print('R ERROR:\n {0}'.format(error.decode("utf-8"))) 
            return None

    except Exception as e: 
        print("dbc2csv - Error converting file: ") 
        print(e)

        return False