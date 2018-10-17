#### Flatiron_School_Final

## Web Scrape:
see: WebS_.py and Sc.py - lines(19 - 60)

While the Starcraft 2 replay files are not found in this repository, The means by which one can retrieve them are. In this script, I use BeautifulSoup, selenium, requests among many other libraries to pull replays from https://gggreplays.com/ and https://lotv.spawningtool.com/. (See import commands at the top of each file for further details)

## Extract Transform Load:
see: ./ORM/ETL_.py and ./ORM/models.py

## Regression
see: ./ORM/PCA_ETL.py

## Unsupervised K-Means - Euclidian:
see: ./ORM/unsupervised.py

## Unsupervised K-Means - Cosine: (in Progress)
see: ./ORM/unsupervised_cos.py

## to-Do
### Dense Neural Network player_state -f-> action:  (in Progress)
see: ML.py and ML_Sc.py and TreeBot.py for initial experiments
### Weighted Vector Space on strategy to construct 'Newtonian Gravitational field' to make decision player_state -f-> action: (in Progress)
see: under construction
### Construct Convolutional data from player A perspective: (in Progress)
see: A way to capture known partial information of Player B Strategy (Estimate player B strategy) adjust own strategy accordingly ie change Weighted Vector Space.
under construction
