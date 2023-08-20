--> Regarding choosing the validation data:

We are taking out a small sample of our target dataset (with trips originating from inside CBD) and using that as our validation dataset on which we will perform hyperparameter tuning. Random selection is done to obtain the validation set, without having any considerations of keeping the mode choice's percentage equal in train and validation set.
To check transferability on a target region, we won't be actually knowing the mode choice distribution in that region, so having a random mode choice in validation set makes more sense, than the mode distribution being similar to our training data.

--------------------------------------------------------------------------------------------------------------

--> The main purpose of transferability:

Let's say we have a significant amount of data for region A, so we can train our mode choice model for region A. Now for region B, we have a very small data sample which is not enough to create a model for region B. Now if we somehow use the small datasample of region B to tune our model made in region A, it will greatly enhance the tranferability of our model to region B, which will further nullify the need to obtain a huge data sample in region B, so saving time and resources.

--------------------------------------------------------------------------------------------------------------

--> The purpose to choose F2 measure as our evaluation metrics:

One of the usecases of this research can be to understand the mode's demand given certain incentives and correspondingly creating an infrastructure which support's that demand.
Now F2 measure gives a balance between precision and recall,
    ``` Lesser the false positives, more the precision```
    ``` Lesser the false negatives, more the recall```
1) Let's assume our model has a lot of false positives, this means it is overestimating each mode's demand, which further means there will be a greater allocation of resources to building infrastructure for a mode, than it is necessary. 
2) Let's assume our model has a lot of false negatives, this means it is underestimating each mode's demand, which further means there won't be enough infrastructure and facilities to support a mode's real demand.
In my opinion the second problem should be a greater concern than the first problem, hence we should use F2 measure which gives more weightable to recall score than to precision score. 
So a model with higher F2 score will try more to suppress the possibility of the 2nd condition than the 1st condition.
(Note: We are not entirely trying to suppress the 2nd condition only, if that was the case we would have choosen recall as our metrics instead of F2 score. F2 score tries to reduce the possibility of both the condition but gives 2nd one more weight.)

---------------------------------------------------------------------------------------------------------------

--> Averaging strategy to compute F2 measure:

To compute F2 measure in a multiclass model, it's generally best to use one over rest strategy to first compute the F2 score of a particular mode with respect to rest of the modes. Once we have computed the F2 score of each mode individually, then we use an averaging strategy to get the model's final F2 score. Averaging strategies can be of the following types:-
1) Macro Averaging :- This strategy assigns equal weight to each mode's score to compute average. This is generally used when the data has class imbalance and we want to compute if the model is truely skilled at predicting other mode's or is it only predicting the majority mode well. (I believe this should be our ideal choice to evaluate the models)

2) Weighted Averaging :- Also sometimes called micro averaging, here unequal weights are assigned to each mode corresponding to their distributing in the dataset. This strategy is useful if the data sample over which evaluation is done, is a good representative of the actual real world mode choice distribution. This strategy will give a higher weight to the majority mode, which will already have a better F2 score than rest of the modes, hence the majority mode will have a really high impact in the final score, and this score will be greater than macro averaging one in majority of the cases. (Suggested by Chikaraishi sensei) 

-------------------------------------------------------------------------------------------------------------