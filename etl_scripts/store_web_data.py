import csv

text_file_path = 'workspace/brennen_langchain/data/url_links/links.txt'

with open(text_file_path, 'r') as file:
    all_urls = []
    for line in file:
        url = line.rstrip().split(',') #using rstrip to remove the \n
        all_urls.extend(url) #using extend instead of append
    all_urls = [elem for elem in all_urls if elem != ""]

