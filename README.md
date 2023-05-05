Download Link: https://assignmentchef.com/product/solved-dsci552-homework-3-time-series-classification
<br>
An interesting task in machine learning is classification of time series. In this problem, we will classify the activities of humans based on time series obtained by a Wireless Sensor Network.

<ul>

 <li>Download the AReM data from: https://archive.ics.uci.edu/ml/datasets/</li>

</ul>

Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM %29 . The dataset contains 7 folders that represent seven types of activities. In each folder, there are multiple files each of which represents an instant of a human performing an activity.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> Each file containis 6 time series collected from activities of the same person, which are called avg rss12, var rss12, avg rss13, var rss13, vg rss23, and ar rss23. There are 88 instances in the dataset, each of which contains 6 time series and each time series has 480 consecutive values.

<ul>

 <li>Keep datasets 1 and 2 in folders bending1 and bending 2, as well as datasets 1, 2, and 3 in other folders as test data and other datasets as train data.</li>

 <li>Feature Extraction</li>

</ul>

Classification of time series usually needs extracting features from them. In this problem, we focus on time-domain features.

<ol>

 <li>Research what types of time-domain features are usually used in time series classification and list them (examples are minimum, maximum, mean, etc).</li>

 <li>Extract the time-domain features minimum, maximum, mean, median, standard deviation, first quartile, and third quartile for all of the 6 time series in each instance. You are free to normalize/standardize features or use them directly.<sup>2</sup></li>

</ol>

Your new dataset will look like this:

<table width="514">

 <tbody>

  <tr>

   <td width="73">Instance</td>

   <td width="49">min<sub>1</sub></td>

   <td width="52">max<sub>1</sub></td>

   <td width="59">mean<sub>1</sub></td>

   <td width="80">median<sub>1</sub></td>

   <td width="37">···</td>

   <td width="77">1st quart<sub>6</sub></td>

   <td width="87">3rd quart<sub>6</sub></td>

  </tr>

  <tr>

   <td width="73">1</td>

   <td width="49"></td>

   <td width="52"></td>

   <td width="59"></td>

   <td width="80"></td>

   <td width="37"></td>

   <td width="77"></td>

   <td width="87"></td>

  </tr>

  <tr>

   <td width="73">2</td>

   <td width="49"></td>

   <td width="52"></td>

   <td width="59"></td>

   <td width="80"></td>

   <td width="37"></td>

   <td width="77"></td>

   <td width="87"></td>

  </tr>

  <tr>

   <td width="73">3</td>

   <td width="49"></td>

   <td width="52"></td>

   <td width="59"></td>

   <td width="80"></td>

   <td width="37"></td>

   <td width="77"></td>

   <td width="87"></td>

  </tr>

  <tr>

   <td width="73">…</td>

   <td width="49">…</td>

   <td width="52">…</td>

   <td width="59">…</td>

   <td width="80">…</td>

   <td width="37">…</td>

   <td width="77">…</td>

   <td width="87">…</td>

  </tr>

  <tr>

   <td width="73">88</td>

   <td width="49"></td>

   <td width="52"></td>

   <td width="59"></td>

   <td width="80"></td>

   <td width="37"></td>

   <td width="77"></td>

   <td width="87"></td>

  </tr>

 </tbody>

</table>

where, for example, 1st quart<sub>6</sub>, means the first quartile of the sixth time series in each of the 88 instances.

<ul>

 <li>Estimate the standard deviation of each of the time-domain features you extracted from the data. Then, use Python’s bootstrapped or any other method to build a 90% bootsrap confidence interval for the standard deviation of each feature.</li>

</ul>

<ol>

 <li>Use your judgement to select the three most important time-domain features (one option may be min, mean, and max).</li>

</ol>

(d) Binary Classification Using Logistic Regression<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

<ol>

 <li>Assume that you want to use the training set to classify bending from other activities, i.e. you have a binary classification problem. Depict scatter plots of the features you specified in 1(c)iv extracted from time series 1, 2, and 6 of each instance, and use color to distinguish bending vs. other activities. (See p. 129 of the textbook).<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a></li>

 <li>Break each time series in your training set into two (approximately) equal length time series. Now instead of 6 time series for each of the training instances, you have 12 time series for each training instance. Repeat the experiment in 1(d)i, i.e depict scatter plots of the features extracted from both parts of the time series 1,2, and 12. Do you see any considerable difference in the results with those of 1(d)i?</li>

</ol>

<ul>

 <li>Break each time series in your training set into <em>l </em>∈ {1<em>,</em>2<em>,…,</em>20} time series of approximately equal length and use logistic regression<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> to solve the binary classification problem, using time-domain features. Remember that breaking each of the time series does not change the number of instances. It only changes the number of features for each instance. Calculate the p-values for your logistic regression parameters in each model corresponding to each value of <em>l </em>and refit a logistic regression model using your pruned set of features.<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>Alternatively, you can use backward selection using sklearn.feature selection or glm in R. Use 5-fold cross-validation to determine the best value of the pair (<em>l,p</em>), where <em>p </em>is the number of features used in recursive feature elimination. Explain what the right way and the wrong way are to perform cross-validation in this problem.<a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a> Obviously, use the right way! Also, you may encounter the problem of class imbalance, which may make some of your folds not having any instances of the rare class. In such a case, you can use <em>stratified cross validation. </em>Research what it means and use it if needed.</li>

</ul>

In the following, you can see an example of applying Python’s Recursive Feature Elimination, which is a backward selection algorithm, to logistic regression.

# Recursive Feature Elimination from sklearn import datasets from sklearn . feature selection import RFE from sklearn . linear model import             LogisticRegression

# load the i r i s datasets dataset = datasets . load iris ()

# create a base c l a s s i f i e r used to evaluate a subset of attributes model = LogisticRegression ()

# create the RFE model and select 3 attributes rfe = RFE(model , 3) rfe = rfe . f i t ( dataset . data , dataset . target )

# summarize the       selection            of           the        attributes print ( rfe . support) print ( rfe . ranking )

<ol>

 <li>Report the confusion matrix and show the ROC and AUC for your classifier on train data. Report the parameters of your logistic regression <em>β<sub>i</sub></em>’s as well as the p-values associated with them.</li>

 <li>Test the classifier on the test set. Remember to break the time series in your test set into the same number of time series into which you broke your training set. Remember that the classifier has to be tested using the features extracted from the test set. Compare the accuracy on the test set with the cross-validation accuracy you obtained previously.</li>

 <li>Do your classes seem to be well-separated to cause instability in calculating logistic regression parameters?</li>

</ol>

<ul>

 <li>From the confusion matrices you obtained, do you see imbalanced classes? If yes, build a logistic regression model based on case-control sampling and adjust its parameters. Report the confusion matrix, ROC, and AUC of the model.</li>

 <li>Binary Classification Using <em>L</em><sub>1</sub>-penalized logistic regression

  <ol>

   <li>Repeat 1(d)iii using <em>L</em><sub>1</sub>-penalized logistic regression,<a href="#_ftn7" name="_ftnref7"><sup>[7]</sup></a> e. instead of using pvalues for variable selection, use <em>L</em><sub>1 </sub>regularization. Note that in this problem, you have to cross-validate for both <em>l</em>, the number of time series into which you break each of your instances, and <em>λ</em>, the weight of <em>L</em><sub>1 </sub>penalty in your logistic regression objective function (or <em>C</em>, the budget). Packages usually perform cross-validation for <em>λ </em>automatically.<sup>9</sup></li>

   <li>Compare the <em>L</em><sub>1</sub>-penalized with variable selection using p-values. Which one performs better? Which one is easier to implement?</li>

  </ol></li>

 <li>Multi-class Classification (The Realistic Case)

  <ol>

   <li>Find the best <em>l </em>in the same way as you found it in 1(e)i to build an <em>L</em><sub>1</sub>penalized multinomial regression model to classify all activities in your training set.<a href="#_ftn8" name="_ftnref8"><sup>[8]</sup></a> Report your test error. Research how confusion matrices and ROC curves are defined for multiclass classification and show them for this problem if possible.<a href="#_ftn9" name="_ftnref9"><sup>[9]</sup></a></li>

   <li>Repeat 1(f)i using a Na¨ıve Bayes’ classifier. Use both Gaussian and Multinomial priors and compare the results. iii. Which method is better for multi-class classification in this problem?</li>

  </ol></li>

</ul>

<ol start="2">

 <li>ISLR 3.7.4</li>

 <li>ISLR, 4.8.3</li>

 <li>ISLR 4.8.7</li>

 <li>Extra Practice (you do not need to submit the answers): ISLR 3.7.3, 3.7.5, 4.8.4, 4.8.9</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> Some of the data files need very minor cleaning. You can do it by Excel or Python. <sup>2</sup>You are welcome to experiment to see if they make a difference.

<a href="#_ftnref2" name="_ftn2">[2]</a> Some logistic regression packages have a built-in <em>L</em><sub>2 </sub>regularization. To remove the effect of <em>L</em><sub>2 </sub>regularization, set <em>λ </em>= 0 or set the budget <em>C </em>→∞ (i.e. a very large value).

<a href="#_ftnref3" name="_ftn3">[3]</a> You are welcome to repeat this experiment with other features as well as with time series 3, 4, and 5 in each instance.

<a href="#_ftnref4" name="_ftn4">[4]</a> If you encountered instability of the logistic regression problem because of linearly separable classes, modify the Max-Iter parameter in logistic regression to stop the algorithm immaturely and prevent from its instability.

<a href="#_ftnref5" name="_ftn5">[5]</a> R calculates the p-values for logistic regression automatically. One way of calculating them in Python is to call R within Python. There are other ways to obtain the p-values as well.

<a href="#_ftnref6" name="_ftn6">[6]</a> This is an interesting problem in which the number of features changes depending on the value of the parameter <em>l </em>that is selected via cross validation. Another example of such a problem is Principal Component Regression, where the number of principal components is selected via cross validation.

<a href="#_ftnref7" name="_ftn7">[7]</a> For <em>L</em><sub>1</sub>-penalized logistic regression, you may want to use normalized/standardized features <sup>9</sup>Using the package Liblinear is strongly recommended.

<a href="#_ftnref8" name="_ftn8">[8]</a> New versions of scikit learn allow using <em>L</em><sub>1</sub>-penalty for multinomial regression.

<a href="#_ftnref9" name="_ftn9">[9]</a> For example, the pROC package in R does the job.