#### Flatiron_School_Final

## Goal:
A study of Starcraft 2 replay data in an attempt to identify strategy exterior to expert knowledge

## Web Scrape:
see: WebS_.py and Sc.py - lines(19 - 60)

While the Starcraft 2 replay files are not found in this repository, The means by which one can retrieve them are. In these scripts, I use *BeautifulSoup*, *selenium* and *requests* among many other libraries to pull replays from https://gggreplays.com/ and https://lotv.spawningtool.com/. (See import commands at the top of each file for further details)

Several interesting challenges were encountered particularly with respect to https://gggreplays.com/ where replay elements were nested in a 3 tiered structure, those being (_Search Page, Content Page, Download Link_). This required the use of selenium webdriver so that the desired elements would be loaded into the browser and be scraped.

In total ~> 120,000 replay files spanning 5 years and 7 leagues, among many other metrics, were collected. For the purposes of this project A subset of replays, particularly those belonging to professional players, were used due not only for the time constraint, but due to the diverse range of player strategy displayed at that level.

## Extract Transform Load:
see: (./ORM/ETL_.py and ./ORM/models.py) or (ETL.py)

Using the Flask-SQLAlchemy python framework a cyclic model (_./ORM/models.py_) of replay elements was constructed:

1. (Players) have many (Events, Games)
2. (Games) have many (Players, Events)
3. (Events) have one (Player, Game)

Ultimately, this construction was a great hinderance to the project. Navigating such a graph was cumbersome at best and confusing at worst. In the light of this burden, I decided to (_post submission_) reconstruct the model in a star topology seen in the current (_./ORM/models.py_).

1. (Users) have many (Participants)
2. (Participants) have many (Events) have one (Game, User)
3. (Games) have many (Participants)
4. (Events) have one (Participant)

This intermediary object (Participant) works to simplify queries and relationships between (Games, Users, Events) and hosts a series of additional variable information from the previous Players class.

With the subset of replays committed to a SQLlite3 database, the raw information was then transformed into a sequential aggregate form.

*ie.*
(game, participant, sequence, train_Marine, train_Marauder, build_Barracks, ...)
1. a_1 = (100, 4, 0, 0, 0, 0, 0, ...)
2. a_2 = (100, 4, 1, 0, 1, 0, 0, ...)
3. a_3 = (100, 4, 2, 1, 0, 0, 0, ...)
4. a_4 = (100, 4, 3, 0, 1, 0, 0, ...)

*into*
(game, participant, action, train_Marine, train_Marauder, build_Barracks, ...)
1. a_1 = (100, 4, 0, 0, 0, 0, 0, ...)
2. a_2 = (100, 4, 1, 0, 1, 0, 0, ...)
3. a_3 = (100, 4, 2, 1, 1, 0, 0, ...)
4. a_4 = (100, 4, 3, 1, 2, 0, 0, ...)

![Image of data](http://oi68.tinypic.com/2wfl0fd.jpg)
_figure above displays all Terran professional games (buildings constructed) notice the clear directionality of the tendrils._

to reflect the current state of the game for one of the two participants. Notice, with (game, participant, action) removed, the bulk can be considered a one dimensional curve in Rn whose rate with respect to order of action belongs to the hypercube Rn and |a_(n)| < |a_(n+m)| for all n and m belong to the Naturals.

## Regression - Singular Vector
see: ./ORM/PCA_ETL.py

For each games sequence of events, I preformed a Principal Component Analysis reduction of dimensionally -> R1 as a means to extract the first singular vector. This, by the definition of the first singular vector [1](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf) (Section 1.1), vector is a regressive representation of the direction of the propagation of events of each game. This was chosen as a representation not only for its speed and interpretability, but for its ability to capture the events for each game in its totality in a single vector. There are certainly disadvantages to this approach with a loss of information (High Bias) and lack of invertibility, but it suited the project goal well enough. I intend to construct a function which measures an 'arc' residual sum as means to interpret the 'goodness of fit' of each singular vector and its corresponding aggregate game events.

![Image of data](http://oi66.tinypic.com/2cpet7r.jpg)
_figure above displays inner product of Terran, Protoss and Zerg professional games as a measure of directional simmilarity. Notice that there are regions of orthogonal singular vectors reflecting different race units and structures._

## Unsupervised K-Means - Euclidian:
see: ./ORM/unsupervised.py

Equipped with our singular vector representation for each game's events, I carried out unsupervised KMeans, with a Euclidian metric and GaussianMixture clustering on these singular vectors. With this algorithm, we were attempting to identify a collection of naturally occurring strategies in the game of Starcraft. I intend on trying several additional methods including a cosine similarity metric, which I believe will parse the singular vectors best according to an adjusted silhouette score metric.  

## Conclusion:

Currently, the only clear assertion I can make is that I will continue to work on this project. Below is a collection of goals in a to-do list for the coming weeks. I did not achieve what I sought out to do in the problem statement of the project, but I am confident that with diligent work I will be able to.

## to-do:
### Unsupervised K-Means - Cosine: (in Progress)
see: ./ORM/unsupervised_cos.py
### Regression ARIMA coefficients / Unsupervised K-Means - Cosine, Euclidian (in Progress)
see: under construction
### Dense Neural Network player_state -f-> action:  (in Progress)
see: ML.py and ML_Sc.py and TreeBot.py
### Weighted Vector Space on strategy to construct 'Newtonian Gravitational field' to make decision player_state -f-> action: (in Progress)
see: under construction
### Construct Convolutional data from player A perspective: (in Progress)
see: A way to capture known partial information of Player B Strategy (Estimate player B strategy) adjust own strategy accordingly ie change Weighted Vector Space.
under construction
