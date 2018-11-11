# onj-hw1

## Results
### Clustering data
Results are available [here](https://github.com/JGasp/onj-hw1/blob/master/results/results.txt).

### Clusters visualization
Numbers near dots represent article index

![clusters](https://raw.githubusercontent.com/JGasp/onj-hw1/master/results/clusters.png)

## Usage
### Building corpus
To manualy download tech news from site [ExtremTech](https://www.extremetech.com/)

You need to download packages :
- selenium (used to render and query Javascript pages)
  - This script uses Chrome as web driver [Chrome WebDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
- unidecode (filters unicode characters)

and run script [download_corpus.py](https://github.com/JGasp/onj-hw1/blob/master/src/download_corpus.py)
Downloaded files then have to be zipped for futher use in analysis script.

### Text analysis
To perform text analysis run script [tech_news_analysis.py](https://github.com/JGasp/onj-hw1/blob/master/src/tech_news_analysis.py)
this is provided you have file [res/tech_news.zip](https://github.com/JGasp/onj-hw1/blob/master/res/tech_news.zip) in working direcotry. The script will produce console output similar to available results and visualize clustering.
