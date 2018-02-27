
# Work Samples from Various Projects

by Miadad Rashid

## Multinomial Artificial Neural Network

For training and in-depth understanding around the math/theory behind the application of **Neural Networks**, I decided to code up from scratch a Multi-Class ANN using images of hands representing numbers.  With a few added functions, the model could be generalized for all output types, not just classification.

[Project Link](https://github.com/thedurphy/Portfolio/blob/master/Deep%20Learning%20Tests/Multinomial%20ANN%20with%20Regularization%20and%20Optimization.ipynb)

## Trading Sample Functions

This fold contains 3 files...
1. [`forexnn.py`](https://github.com/thedurphy/Portfolio/blob/master/Trading%20Sample%20Functions/forexnn.py)
This file contains wrapper functions I made to streamline use of `keras` and `tensorflow` libraries for use with the Oanda API.  These funtions include methods for creating sequential stacks for recurrent neural networks, normalization methods for financial data, de-trending time-series data for analysis, etc.
2. [`indicators.py`](https://github.com/thedurphy/Portfolio/blob/master/Trading%20Sample%20Functions/indicators.py)
This file contains several common indicators used in Technical Analysis Trading of exchanges (Exponential Moving Averages, Digital Filters, Boilinger Bands, Heikin Ashi Candles, and even a personally made indicator called a Neural Net Indicator which uses the predictions of a neural net on price movement as an indicator for a separate neural net.
3. [`oandafunctions.py`](https://github.com/thedurphy/Portfolio/blob/master/Trading%20Sample%20Functions/oandafunctions.py)
This file contains wrappers I made for the `oandapyV20` library to streamline certain methods in the library.

## Synergy Coefficient

This was an interesting project for UHG that I am very proud of.  The objective was to combine the best mentors and mentees along with creating the most productive teams of executive and employees.  All we had was profile data for all the participants that were gathered by several different means of data mining, etc.  Using all the different attributes, I created a series of metrics to define how individuals and groups would likely work together.  The research paper is contain at the following link.  Please note that certain information has been stripped (blacked out) for confidentiality purposes.

[Project Link](https://htmlpreview.github.io/?https://github.com/thedurphy/Portfolio/blob/master/UHN%20NML%20Reasoning.html)

## Boston Housing Project
This was a course final project for Udacity that I helped design and test.  Basically, we walk through with the student designing a decision tree to predict housing prices for the Boston region.  We then go through the concepts of cross-validation and model health/performances touching on the bias/variance trade-off dilema in basic machine learning.

[Project Link](https://github.com/thedurphy/Portfolio/blob/master/boston_housing.pdf)

[Code Link](https://github.com/thedurphy/Portfolio/blob/master/boston_housing.py)

## Human Activity Recognition
This was one of my very first project dealing with data gathered from participants cellphone's gyroscopic and accelerometer data.  The objective was to identify their physical activity (sleeping, walking, running, stairs, etc.) by the readings between these sensors.  It was very cool in illustrating the power of machine learning (if you notice, I achieved 100% accuracy on the validation dataset).  This was also one of my projects done in R.

[Project Link](https://htmlpreview.github.io/?https://github.com/thedurphy/Portfolio/blob/master/pmlProject.html)

## Open Street Maps ETL
This is a project for Open Street Map where we extracted their XML data, corrected issues, and loaded it to JSON formatting.  For this project, I went a little over board and cross-validated data on OSM with data from Google Maps and automated corrections accordingly.  I also added a little bit of analysis at the end to identify individuals using bots.

[Project Link](https://github.com/thedurphy/Portfolio/blob/master/osmproject.pdf)

[Code Link](https://github.com/thedurphy/Portfolio/blob/master/osmproject.py)

## EIT Health Summit
This was a fairly recent project for EIT Health located in Germany.  The project was more of the social-engineering concepts used with the **Synergy Coefficient** study above.  The objective was to extract the data, clean the data, and get the data to the show meaningful trends in the participants network data.  From their email addresses, LinkedIn, and other public information, we identify their personality traits.  From the personality traits and skills mentioned in LinkedIn, we optimize seating charts and gathering parameters to increase the probability of people meeting that have aligned goals.  Below is a small code snippet from the data extraction and finalization.  Unfortunately, I cannot share the code for the sentiment/text analysis I made to extrapolate personality types.

[Project Link](https://github.com/thedurphy/Portfolio/blob/master/EIT.ipynb)

[Network Maps](https://github.com/thedurphy/Portfolio/blob/master/eit2018graphs.pdf)

## Lis Miserables Network of Characters
This was inspired by a fellow network grapher.  It is basically a visualization of all the characters in Les Miserables and their network connections.  It is very easy to identify key characters through such a simple presentation.

[Visualization Link](https://htmlpreview.github.io/?https://github.com/thedurphy/Portfolio/blob/master/LesMiserables.html)

[Code Link](https://htmlpreview.github.io/?https://github.com/thedurphy/Portfolio/blob/master/lesmis_example.html)
