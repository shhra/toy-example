## Problem Formulation 

**Task**

		We want a model that can predict the emotion as well 
		as the sentiment in the sentence 
		so that it can be used to visualize the model.

**Experience** 

		Using a corpus of sentences (social media feeds) that have 
		a defined emotion as labels, 
		we can train a language model to perform classification.

**Performance**

		As measured by classification accuracy, 

**Success Criteria**

		It’s a success if the model can group tweets based on a product 
		and visualize the amount of social media feeds
		for different emotions about the product.

**Other metric of interest are**

		We may also be interested in the speed at which
		the algorithm classifies the sentence. 
		If the algorithm is slow then it wouldn’t be beneficial to us.

## Solution Formulation 

**Manually, the problem could be solved as**

		Number of positive/negative product reviews from comments and user feedbacks

**It can be formulated as a ML problem as**

		Classification

**A similar ML task is**

		Clustering task as fear and surprise can be clustered in the same group,
		anger and disgust can be clustered in the same group .
		Similar to other Gross sounding words.

**Our assumption are** 
1. The specific words used in the tweet/feeds matter to the model.
2. The number of retweets/share may matter to the model.
3. Older tweets/feeds are less predictive than more recent tweets/feeds.

**A baseline approach could be**

		Treat as Classification; 
		Naive bayes classifier

