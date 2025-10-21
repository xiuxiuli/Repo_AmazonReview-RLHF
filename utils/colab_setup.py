from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/xiuxiuli/Repo_AmazonReview-RLHF.git /content/drive/MyDrive/amazon_review_RLHF

%cd /content/drive/MyDrive/amazon_review_RLHF

!git pull

!pip install -r requirements.txt

!python ./data/download.py

