function estimateGiniImpurity(feature_values, threshold, labels, polarity)
    if count(feat -> polarity(feat, threshold), feature_values) == 0
        return 1
    end
    
    all_labels = unique(labels)
    pr_classes  = map(
        gini_label -> length([
                feature for (label, feature)
                in zip(labels, feature_values)
                if polarity(feature, threshold) && label == gini_label
        ]) / count(feat -> polarity(feat, threshold), feature_values),
        all_labels
    )

    sum(map(pr_class -> pr_class * (1 - pr_class), pr_classes))
end

function estimateGiniImpurityExpectation(feature_values, threshold, labels)
   num_datapoints = length(feature_values) 

    return sum(
        (sum(op.(feature_values, threshold) / num_datapoints) for op in [(a, b) -> a <= b, (a, b) -> a >= b]) .* 
        estimateGiniImpurity(feature_values, threshold, labels, op)
        for op in [(a, b) -> a <= b, (a, b) -> a >= b]
    )
end




feature_values = [1,2,3,4,5,6,7,8]
labels = [+1,+1,+1,+1, -1,-1,-1,-1]


for threshold = 0:8
    println(estimateGiniImpurity(feature_values, threshold, labels, (a, b) -> a >= b))
end
