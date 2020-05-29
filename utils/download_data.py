import os
import subprocess

if os.getenv("SERVER") == '0':
    def __check_data_folder__():
        """Function that check if in content folder exists the data folder, otherwise is created."""
        if os.path.isdir("/content/data"):
            os.mkdir("/content/data")
        
    def boston201(file_name:str = "boston201.zip"):
        """Function to download and unzip the dataset Boston201 from drive in the data 
        content folder. The Dataset includes the jpg, ppm and background substraction videos 
        in it."""
        __check_data_folder__()
        file_id = "1Ph_Ys3O_vI93WeTkDqr5h6kTJm0CZ0Ub"
        subprocess.run(["bash", "/content/utils/download_from_drive.sh", 
            file_id, file_name], check=True)
        subprocess.run(["unzip", "-q", file_name], check=True)
        subprocess.run(["rm", file_name], check=True)
        subprocess.run(["mv", "boston201", "data/boston201"], check=True)

    def embedding_word_vectors(file_name:str = "word_vectors.zip"):
        """Function to download and unzip the embedding words vectors from drive in the data 
        content folder."""
        __check_data_folder__()
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
        subprocess.run(["wget", "-q", url, "-O", file_name], check=True)
        subprocess.run(["unzip", "-q", file_name], check=True)
        subprocess.run(["rm", "-rf", file_name], check=True)
        subprocess.run(["mv", "wiki-news-300d-1M.vec", "data/wiki-news-300d-1M.vec"], check=True)

elif os.getenv("SERVER") == '1':
    raise NotImplementedError("This module is not intended to work in the local "
        "server because the data is already in it. Just find it ;)")

else:
    raise ImportError("You can't import the download_data module because you "
        "didn't set up the env variable 'SERVER'")