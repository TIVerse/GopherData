// Package tree provides decision tree algorithms.
package tree

import (
	"fmt"
	"math"
	"sort"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// DecisionTree implements a decision tree for classification and regression.
type DecisionTree struct {
	// MaxDepth is the maximum depth of the tree (0 = unlimited)
	MaxDepth int
	
	// MinSamples is the minimum samples required to split a node
	MinSamples int
	
	// Criterion for splitting: "gini", "entropy" (classification), "mse" (regression)
	Criterion string
	
	// root is the root node of the tree
	root *treeNode
	
	// fitted indicates whether the model has been fitted
	fitted bool
	
	// isClassification indicates if this is a classification tree
	isClassification bool
	
	// classes stores unique class labels (for classification)
	classes []string
	
	// featureImportances stores the importance of each feature
	featureImportances []float64
}

// treeNode represents a node in the decision tree.
type treeNode struct {
	// feature is the feature index used for splitting (-1 for leaf)
	feature int
	
	// threshold is the threshold value for the split
	threshold float64
	
	// left and right children
	left  *treeNode
	right *treeNode
	
	// value is the predicted value for leaf nodes
	value any
	
	// samples is the number of samples at this node
	samples int
	
	// impurity is the impurity measure at this node
	impurity float64
}

// NewDecisionTreeClassifier creates a new decision tree for classification.
func NewDecisionTreeClassifier(maxDepth, minSamples int, criterion string) *DecisionTree {
	if maxDepth <= 0 {
		maxDepth = 10
	}
	if minSamples < 2 {
		minSamples = 2
	}
	if criterion == "" {
		criterion = "gini"
	}
	
	return &DecisionTree{
		MaxDepth:         maxDepth,
		MinSamples:       minSamples,
		Criterion:        criterion,
		isClassification: true,
		fitted:           false,
	}
}

// NewDecisionTreeRegressor creates a new decision tree for regression.
func NewDecisionTreeRegressor(maxDepth, minSamples int) *DecisionTree {
	if maxDepth <= 0 {
		maxDepth = 10
	}
	if minSamples < 2 {
		minSamples = 2
	}
	
	return &DecisionTree{
		MaxDepth:         maxDepth,
		MinSamples:       minSamples,
		Criterion:        "mse",
		isClassification: false,
		fitted:           false,
	}
}

// Fit trains the decision tree.
func (dt *DecisionTree) Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error {
	// Extract features and target
	features, err := extractFeaturesTree(X)
	if err != nil {
		return err
	}
	
	target := make([]any, y.Len())
	for i := 0; i < y.Len(); i++ {
		val, ok := y.Get(i)
		if !ok || val == nil {
			return fmt.Errorf("target contains null at index %d", i)
		}
		target[i] = val
	}
	
	if len(features) != len(target) {
		return fmt.Errorf("X and y must have same length")
	}
	
	// For classification, extract classes
	if dt.isClassification {
		classSet := make(map[string]bool)
		for _, val := range target {
			classSet[fmt.Sprint(val)] = true
		}
		dt.classes = make([]string, 0, len(classSet))
		for class := range classSet {
			dt.classes = append(dt.classes, class)
		}
	}
	
	// Build tree
	indices := make([]int, len(features))
	for i := range indices {
		indices[i] = i
	}
	
	dt.root = dt.buildTree(features, target, indices, 0)
	
	// Calculate feature importances
	nFeatures := len(features[0])
	dt.featureImportances = make([]float64, nFeatures)
	dt.calculateImportances(dt.root, len(features))
	
	dt.fitted = true
	return nil
}

// buildTree recursively builds the decision tree.
func (dt *DecisionTree) buildTree(features [][]float64, target []any, indices []int, depth int) *treeNode {
	node := &treeNode{
		feature:  -1,
		samples:  len(indices),
		impurity: dt.calculateImpurity(target, indices),
	}
	
	// Stopping criteria
	if depth >= dt.MaxDepth || len(indices) < dt.MinSamples {
		node.value = dt.leafValue(target, indices)
		return node
	}
	
	// Check if all samples have same target
	if dt.isHomogeneous(target, indices) {
		node.value = dt.leafValue(target, indices)
		return node
	}
	
	// Find best split
	bestFeature, bestThreshold, bestGain := dt.findBestSplit(features, target, indices)
	
	if bestGain <= 0 {
		node.value = dt.leafValue(target, indices)
		return node
	}
	
	// Split data
	leftIndices, rightIndices := dt.splitData(features, indices, bestFeature, bestThreshold)
	
	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		node.value = dt.leafValue(target, indices)
		return node
	}
	
	// Create split node
	node.feature = bestFeature
	node.threshold = bestThreshold
	node.left = dt.buildTree(features, target, leftIndices, depth+1)
	node.right = dt.buildTree(features, target, rightIndices, depth+1)
	
	return node
}

// findBestSplit finds the best feature and threshold to split on.
func (dt *DecisionTree) findBestSplit(features [][]float64, target []any, indices []int) (int, float64, float64) {
	bestFeature := -1
	bestThreshold := 0.0
	bestGain := -math.MaxFloat64
	
	nFeatures := len(features[0])
	currentImpurity := dt.calculateImpurity(target, indices)
	
	for feature := 0; feature < nFeatures; feature++ {
		// Get unique values for this feature
		values := make([]float64, len(indices))
		for i, idx := range indices {
			values[i] = features[idx][feature]
		}
		
		uniqueValues := getUniqueSorted(values)
		
		// Try each threshold
		for i := 0; i < len(uniqueValues)-1; i++ {
			threshold := (uniqueValues[i] + uniqueValues[i+1]) / 2
			
			leftIndices, rightIndices := dt.splitData(features, indices, feature, threshold)
			
			if len(leftIndices) == 0 || len(rightIndices) == 0 {
				continue
			}
			
			// Calculate information gain
			leftImpurity := dt.calculateImpurity(target, leftIndices)
			rightImpurity := dt.calculateImpurity(target, rightIndices)
			
			n := float64(len(indices))
			nLeft := float64(len(leftIndices))
			nRight := float64(len(rightIndices))
			
			gain := currentImpurity - (nLeft/n)*leftImpurity - (nRight/n)*rightImpurity
			
			if gain > bestGain {
				bestGain = gain
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}
	
	return bestFeature, bestThreshold, bestGain
}

// calculateImpurity calculates the impurity measure.
func (dt *DecisionTree) calculateImpurity(target []any, indices []int) float64 {
	if len(indices) == 0 {
		return 0
	}
	
	if dt.isClassification {
		switch dt.Criterion {
		case "gini":
			return dt.giniImpurity(target, indices)
		case "entropy":
			return dt.entropyImpurity(target, indices)
		default:
			return dt.giniImpurity(target, indices)
		}
	} else {
		return dt.mseImpurity(target, indices)
	}
}

// giniImpurity calculates Gini impurity for classification.
func (dt *DecisionTree) giniImpurity(target []any, indices []int) float64 {
	counts := make(map[string]int)
	for _, idx := range indices {
		label := fmt.Sprint(target[idx])
		counts[label]++
	}
	
	n := float64(len(indices))
	gini := 1.0
	
	for _, count := range counts {
		p := float64(count) / n
		gini -= p * p
	}
	
	return gini
}

// entropyImpurity calculates entropy for classification.
func (dt *DecisionTree) entropyImpurity(target []any, indices []int) float64 {
	counts := make(map[string]int)
	for _, idx := range indices {
		label := fmt.Sprint(target[idx])
		counts[label]++
	}
	
	n := float64(len(indices))
	entropy := 0.0
	
	for _, count := range counts {
		if count > 0 {
			p := float64(count) / n
			entropy -= p * math.Log2(p)
		}
	}
	
	return entropy
}

// mseImpurity calculates MSE for regression.
func (dt *DecisionTree) mseImpurity(target []any, indices []int) float64 {
	if len(indices) == 0 {
		return 0
	}
	
	// Calculate mean
	sum := 0.0
	for _, idx := range indices {
		sum += toFloat64Tree(target[idx])
	}
	mean := sum / float64(len(indices))
	
	// Calculate MSE
	mse := 0.0
	for _, idx := range indices {
		diff := toFloat64Tree(target[idx]) - mean
		mse += diff * diff
	}
	
	return mse / float64(len(indices))
}

// splitData splits indices based on feature and threshold.
func (dt *DecisionTree) splitData(features [][]float64, indices []int, feature int, threshold float64) ([]int, []int) {
	left := make([]int, 0)
	right := make([]int, 0)
	
	for _, idx := range indices {
		if features[idx][feature] <= threshold {
			left = append(left, idx)
		} else {
			right = append(right, idx)
		}
	}
	
	return left, right
}

// isHomogeneous checks if all samples have the same target value.
func (dt *DecisionTree) isHomogeneous(target []any, indices []int) bool {
	if len(indices) <= 1 {
		return true
	}
	
	first := fmt.Sprint(target[indices[0]])
	for _, idx := range indices[1:] {
		if fmt.Sprint(target[idx]) != first {
			return false
		}
	}
	return true
}

// leafValue determines the prediction value for a leaf node.
func (dt *DecisionTree) leafValue(target []any, indices []int) any {
	if dt.isClassification {
		// Return most common class
		counts := make(map[string]int)
		for _, idx := range indices {
			label := fmt.Sprint(target[idx])
			counts[label]++
		}
		
		maxCount := 0
		var mostCommon string
		for label, count := range counts {
			if count > maxCount {
				maxCount = count
				mostCommon = label
			}
		}
		return mostCommon
	} else {
		// Return mean for regression
		sum := 0.0
		for _, idx := range indices {
			sum += toFloat64Tree(target[idx])
		}
		return sum / float64(len(indices))
	}
}

// Predict makes predictions on new data.
func (dt *DecisionTree) Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	if !dt.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	features, err := extractFeaturesTree(X)
	if err != nil {
		return nil, err
	}
	
	predictions := make([]any, len(features))
	for i, sample := range features {
		predictions[i] = dt.predictSample(dt.root, sample)
	}
	
	dtype := core.DtypeString
	if !dt.isClassification {
		dtype = core.DtypeFloat64
	}
	
	return seriesPkg.New("predictions", predictions, dtype), nil
}

// predictSample predicts for a single sample.
func (dt *DecisionTree) predictSample(node *treeNode, sample []float64) any {
	if node.feature == -1 {
		return node.value
	}
	
	if sample[node.feature] <= node.threshold {
		return dt.predictSample(node.left, sample)
	}
	return dt.predictSample(node.right, sample)
}

// FeatureImportances returns the importance of each feature.
func (dt *DecisionTree) FeatureImportances() []float64 {
	return dt.featureImportances
}

// calculateImportances calculates feature importances based on impurity decrease.
func (dt *DecisionTree) calculateImportances(node *treeNode, totalSamples int) {
	if node == nil || node.feature == -1 {
		return
	}
	
	// Importance = (samples / total) * impurity_decrease
	leftImpurity := 0.0
	rightImpurity := 0.0
	leftSamples := 0
	rightSamples := 0
	
	if node.left != nil {
		leftImpurity = node.left.impurity
		leftSamples = node.left.samples
	}
	if node.right != nil {
		rightImpurity = node.right.impurity
		rightSamples = node.right.samples
	}
	
	importance := float64(node.samples) / float64(totalSamples)
	importance *= node.impurity - 
		(float64(leftSamples)/float64(node.samples))*leftImpurity -
		(float64(rightSamples)/float64(node.samples))*rightImpurity
	
	dt.featureImportances[node.feature] += importance
	
	dt.calculateImportances(node.left, totalSamples)
	dt.calculateImportances(node.right, totalSamples)
}

// Helper functions

func extractFeaturesTree(X *dataframe.DataFrame) ([][]float64, error) {
	n := X.Nrows()
	cols := X.Columns()
	
	numericCols := make([]string, 0)
	for _, col := range cols {
		series, err := X.Column(col)
		if err != nil {
			continue
		}
		if isNumericTree(series.Dtype()) {
			numericCols = append(numericCols, col)
		}
	}
	
	if len(numericCols) == 0 {
		return nil, fmt.Errorf("no numeric columns found")
	}
	
	features := make([][]float64, n)
	for i := range features {
		features[i] = make([]float64, len(numericCols))
	}
	
	for j, col := range numericCols {
		series, _ := X.Column(col)
		for i := 0; i < n; i++ {
			val, ok := series.Get(i)
			if ok && val != nil {
				features[i][j] = toFloat64Tree(val)
			}
		}
	}
	
	return features, nil
}

func isNumericTree(dtype core.Dtype) bool {
	return dtype == core.DtypeFloat64 || dtype == core.DtypeInt64
}

func toFloat64Tree(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return 0
	}
}

func getUniqueSorted(values []float64) []float64 {
	uniqueMap := make(map[float64]bool)
	for _, v := range values {
		uniqueMap[v] = true
	}
	
	unique := make([]float64, 0, len(uniqueMap))
	for v := range uniqueMap {
		unique = append(unique, v)
	}
	
	sort.Float64s(unique)
	return unique
}
