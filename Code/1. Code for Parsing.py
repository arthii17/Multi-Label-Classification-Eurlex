# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import urllib.request as ur
from bs4 import BeautifulSoup

filenames = ['21961A1209(01)_EN_NOT.html','21964A1229(01)_EN_NOT.html']


filename = 'C:\\Users\\Mahantesh\\Downloads\\eurlex_html_EN_NOT\\21964A1229(01)_EN_NOT.html'
f = open(filename, "r").read()

#store the labels of the documents
final_results =[]


#iterate over documents
for i in range(len(filenames)):
    file=''
    file = 'C:\\Users\\Mahantesh\\Downloads\\eurlex_html_EN_NOT\\' + str(filenames[i])
    f = open(file, "r").read()
    #print (f)
    soup = BeautifulSoup(f, 'html.parser')
    #print (soup)
    # find_all() will help to fetch all the details of the ahref
    links = soup.find_all('a')
    #collect all href
    pd =[]
    for link in links:
        pd.append(link.get('href'))


    #Extract the Labels
    result =[]

    for i in range(len(pd)):
        print (i)
        if(type(pd[i])==str):
        #Filter for href having Eurovoc
            if(pd[i].find("EUROVOC")!=-1):
                index=0
                length=0
                index=pd[i].rfind("EUROVOC")
                length = len(pd[i])
                result.append(pd[i][index:length])
    #store the labels of every document
    final_results.append(result)
            
            








# Take out the <div> of name and get its value
name_box = soup.find('ul')

print(name_box)