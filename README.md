# Assignment2SOC
Social media network analysis github repository
# Samsung vs iPhone YouTube Sentiment and Network Analysis

## Project Overview

This repository contains the code and outputs for Assignment 2 in Social Media and Network Analysis.

The project analyses YouTube discussions about Samsung and iPhone smartphones. It uses YouTube comments and reply data to examine user sentiment, recurring discussion topics, and interaction patterns between users.

The project combines:

- YouTube Data API collection
- Comment cleaning and preprocessing
- Brand-based comment classification
- VADER sentiment analysis
- TF-IDF keyword extraction
- LDA topic modelling
- NetworkX graph modelling and centrality analysis

## Research Question

How do YouTube users discuss Samsung and iPhone, and which brand receives stronger sentiment, more engagement, and more influential discussion patterns in comment networks?

## Data Source

Data was collected using the YouTube Data API v3.
How to run the codes:
1. Install dependencies
pip install -r reuirements.txt
2. Add YouTube API key
Create a local .env file, inside YOUTUBE_API_KEY=your_api_key_here add this and replace it with ur api key after "=".
3. Run scripts in order
collectyoutubedata.py>cleandata.py>sentimentanalysis.py>keywordtopicanalysis.py>networkanalysis.py
Generated tables are store in outputs/tables/
Generated figures are stored in outputs/figures/
Authors
SM Tasin (S3991074)
Gaurav Shiven Sursen (S3941388)
Jimmy Shankil
Undergraduate Group 71
RMIT University
