# Analysis of the Circle clustering algorithm
<table>
<tr>
<td>
  In this repo are shown the analysis of a new kind of clustering algorithm.

  The analysis compares the kmeans clustering to the new circle clustering algorithm using 
    
    - silhoutte
    - calinski_harabasz
    - dunn
    - pearson
    - average within cluster dissimilarities
    - separation_index
    - entropy
    - widest within cluster gap
    - prediction_strength
    - clustering validity index based on nearest neighbours

  Also are shown the results for the circle clustering using euclidean distance and cosine distance

</td>
</tr>
</table>

### Legend
![](/images_README/legend.png)

## Database 1
![](/images_README/db1.png)
### Euclidean distance
![](/images_README/res_euclidean/euclidean_db1.png)
### Cosine distance
![](/images_README/res_cosine/cosine_db1.png)

## Database 2
![](/images_README/db2.png)
### Euclidean distance
![](/images_README/res_euclidean/euclidean_db2.png)
### Cosine distance
![](/images_README/res_cosine/cosine_db2.png)

## Database 3
![](/images_README/db3.png)
### Euclidean distance
![](/images_README/res_euclidean/euclidean_db3.png)
### Cosine distance
![](/images_README/res_cosine/cosine_db3.png)

## Database 4
![](/images_README/db4.png)
### Euclidean distance
![](/images_README/res_euclidean/euclidean_db4.png)
### Cosine distance
![](/images_README/res_cosine/cosine_db4.png)

## Database 5
![](/images_README/db5.png)
### Euclidean distance
![](/images_README/res_euclidean/euclidean_db5.png)
### Cosine distance
![](/images_README/res_cosine/cosine_db5.png)

## Database 6
![](/images_README/db6.png)
### Euclidean distance
![](/images_README/res_euclidean/euclidean_db6.png)
### Cosine distance
![](/images_README/res_cosine/cosine_db6.png)






## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. 
### Prerequisites

Requirements for the software and other tools to build, test and push 
- git
- Python3 
- Python env installed

### Installing

Downloading the repo

    git clone https://github.com/MorganMalavasi/clustersFinder.git

Enter the directory "k_detection" and run the command to activate the python env (Mac Os)

    source bin/activate

For other OS see the documenatation 
- [Link](https://docs.python.org/3/library/venv.html)


## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

To execute the test on the local machine enter in the src directory and run the command

    python3 main.py

### For modifying the type of test see the source file src/data_generation.py

## Authors

  - **Morgan Malavasi**
