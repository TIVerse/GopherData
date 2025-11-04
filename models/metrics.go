package models

import (
	"fmt"
	"math"

	seriesPkg "github.com/TIVerse/GopherData/series"
)

// ClassificationReport contains precision, recall, F1, and support for each class.
type ClassificationReport struct {
	Precision map[string]float64
	Recall    map[string]float64
	F1Score   map[string]float64
	Support   map[string]int
}

// Accuracy calculates the accuracy score (correct predictions / total predictions).
func Accuracy(yTrue, yPred *seriesPkg.Series[any]) float64 {
	if yTrue.Len() != yPred.Len() {
		return 0
	}
	
	correct := 0
	total := 0
	
	for i := 0; i < yTrue.Len(); i++ {
		trueVal, ok1 := yTrue.Get(i)
		predVal, ok2 := yPred.Get(i)
		
		if !ok1 || !ok2 || trueVal == nil || predVal == nil {
			continue
		}
		
		total++
		if fmt.Sprint(trueVal) == fmt.Sprint(predVal) {
			correct++
		}
	}
	
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total)
}

// Precision calculates the precision score.
// average: "binary" (for binary classification), "micro", "macro", "weighted"
func Precision(yTrue, yPred *seriesPkg.Series[any], average string) float64 {
	cm := confusionMatrixMap(yTrue, yPred)
	
	switch average {
	case "binary":
		return binaryPrecision(cm)
	case "micro":
		return microPrecision(cm)
	case "macro":
		return macroPrecision(cm)
	case "weighted":
		return weightedPrecision(cm, yTrue)
	default:
		return macroPrecision(cm)
	}
}

// Recall calculates the recall score.
func Recall(yTrue, yPred *seriesPkg.Series[any], average string) float64 {
	cm := confusionMatrixMap(yTrue, yPred)
	
	switch average {
	case "binary":
		return binaryRecall(cm)
	case "micro":
		return microRecall(cm)
	case "macro":
		return macroRecall(cm)
	case "weighted":
		return weightedRecall(cm, yTrue)
	default:
		return macroRecall(cm)
	}
}

// F1Score calculates the F1 score (harmonic mean of precision and recall).
func F1Score(yTrue, yPred *seriesPkg.Series[any], average string) float64 {
	p := Precision(yTrue, yPred, average)
	r := Recall(yTrue, yPred, average)
	
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// ConfusionMatrix computes the confusion matrix.
// Returns a 2D slice where cm[i][j] is the count of samples with true label i and predicted label j.
func ConfusionMatrix(yTrue, yPred *seriesPkg.Series[any]) [][]int {
	// Get unique labels
	labels := getUniqueLabels(yTrue, yPred)
	n := len(labels)
	
	// Create label to index mapping
	labelIdx := make(map[string]int)
	for i, label := range labels {
		labelIdx[label] = i
	}
	
	// Initialize confusion matrix
	cm := make([][]int, n)
	for i := range cm {
		cm[i] = make([]int, n)
	}
	
	// Fill confusion matrix
	for i := 0; i < yTrue.Len(); i++ {
		trueVal, ok1 := yTrue.Get(i)
		predVal, ok2 := yPred.Get(i)
		
		if !ok1 || !ok2 || trueVal == nil || predVal == nil {
			continue
		}
		
		trueLabel := fmt.Sprint(trueVal)
		predLabel := fmt.Sprint(predVal)
		
		if trueIdx, ok := labelIdx[trueLabel]; ok {
			if predIdx, ok2 := labelIdx[predLabel]; ok2 {
				cm[trueIdx][predIdx]++
			}
		}
	}
	
	return cm
}

// ClassificationReportFunc generates a comprehensive classification report.
func ClassificationReportFunc(yTrue, yPred *seriesPkg.Series[any]) ClassificationReport {
	labels := getUniqueLabels(yTrue, yPred)
	cm := confusionMatrixMap(yTrue, yPred)
	
	report := ClassificationReport{
		Precision: make(map[string]float64),
		Recall:    make(map[string]float64),
		F1Score:   make(map[string]float64),
		Support:   make(map[string]int),
	}
	
	for _, label := range labels {
		tp := cm[label][label]
		
		// Calculate FP
		fp := 0
		for _, predLabel := range labels {
			if predLabel != label {
				fp += cm[predLabel][label]
			}
		}
		
		// Calculate FN
		fn := 0
		for _, trueLabel := range labels {
			if trueLabel != label {
				fn += cm[label][trueLabel]
			}
		}
		
		// Support
		support := 0
		for _, trueLabel := range labels {
			support += cm[label][trueLabel]
		}
		report.Support[label] = support
		
		// Precision
		if tp+fp > 0 {
			report.Precision[label] = float64(tp) / float64(tp+fp)
		}
		
		// Recall
		if tp+fn > 0 {
			report.Recall[label] = float64(tp) / float64(tp+fn)
		}
		
		// F1
		p := report.Precision[label]
		r := report.Recall[label]
		if p+r > 0 {
			report.F1Score[label] = 2 * p * r / (p + r)
		}
	}
	
	return report
}

// MSE calculates the Mean Squared Error.
func MSE(yTrue, yPred *seriesPkg.Series[any]) float64 {
	if yTrue.Len() != yPred.Len() {
		return math.NaN()
	}
	
	sumSq := 0.0
	count := 0
	
	for i := 0; i < yTrue.Len(); i++ {
		trueVal, ok1 := yTrue.Get(i)
		predVal, ok2 := yPred.Get(i)
		
		if !ok1 || !ok2 || trueVal == nil || predVal == nil {
			continue
		}
		
		trueFloat := toFloat64Metrics(trueVal)
		predFloat := toFloat64Metrics(predVal)
		
		diff := trueFloat - predFloat
		sumSq += diff * diff
		count++
	}
	
	if count == 0 {
		return math.NaN()
	}
	return sumSq / float64(count)
}

// RMSE calculates the Root Mean Squared Error.
func RMSE(yTrue, yPred *seriesPkg.Series[any]) float64 {
	return math.Sqrt(MSE(yTrue, yPred))
}

// MAE calculates the Mean Absolute Error.
func MAE(yTrue, yPred *seriesPkg.Series[any]) float64 {
	if yTrue.Len() != yPred.Len() {
		return math.NaN()
	}
	
	sumAbs := 0.0
	count := 0
	
	for i := 0; i < yTrue.Len(); i++ {
		trueVal, ok1 := yTrue.Get(i)
		predVal, ok2 := yPred.Get(i)
		
		if !ok1 || !ok2 || trueVal == nil || predVal == nil {
			continue
		}
		
		trueFloat := toFloat64Metrics(trueVal)
		predFloat := toFloat64Metrics(predVal)
		
		sumAbs += math.Abs(trueFloat - predFloat)
		count++
	}
	
	if count == 0 {
		return math.NaN()
	}
	return sumAbs / float64(count)
}

// R2Score calculates the R² coefficient of determination.
func R2Score(yTrue, yPred *seriesPkg.Series[any]) float64 {
	// Calculate mean of true values
	var sum float64
	var count int
	for i := 0; i < yTrue.Len(); i++ {
		val, ok := yTrue.Get(i)
		if ok && val != nil {
			sum += toFloat64Metrics(val)
			count++
		}
	}
	if count == 0 {
		return math.NaN()
	}
	mean := sum / float64(count)
	
	// Calculate SS_res and SS_tot
	var ssRes, ssTot float64
	for i := 0; i < yTrue.Len(); i++ {
		trueVal, ok1 := yTrue.Get(i)
		predVal, ok2 := yPred.Get(i)
		
		if !ok1 || !ok2 || trueVal == nil || predVal == nil {
			continue
		}
		
		trueFloat := toFloat64Metrics(trueVal)
		predFloat := toFloat64Metrics(predVal)
		
		diffRes := trueFloat - predFloat
		ssRes += diffRes * diffRes
		diffTot := trueFloat - mean
		ssTot += diffTot * diffTot
	}
	
	if ssTot == 0 {
		return math.NaN()
	}
	return 1 - (ssRes / ssTot)
}

// AdjustedR2 calculates the adjusted R² score.
func AdjustedR2(yTrue, yPred *seriesPkg.Series[any], nFeatures int) float64 {
	r2 := R2Score(yTrue, yPred)
	n := yTrue.Len()
	p := nFeatures
	
	if n-p-1 <= 0 {
		return math.NaN()
	}
	
	return 1 - ((1 - r2) * float64(n-1) / float64(n-p-1))
}

// Helper functions

func confusionMatrixMap(yTrue, yPred *seriesPkg.Series[any]) map[string]map[string]int {
	cm := make(map[string]map[string]int)
	
	for i := 0; i < yTrue.Len(); i++ {
		trueVal, ok1 := yTrue.Get(i)
		predVal, ok2 := yPred.Get(i)
		
		if !ok1 || !ok2 || trueVal == nil || predVal == nil {
			continue
		}
		
		trueLabel := fmt.Sprint(trueVal)
		predLabel := fmt.Sprint(predVal)
		
		if cm[trueLabel] == nil {
			cm[trueLabel] = make(map[string]int)
		}
		cm[trueLabel][predLabel]++
	}
	
	return cm
}

func getUniqueLabels(series ...*seriesPkg.Series[any]) []string {
	labelSet := make(map[string]bool)
	
	for _, s := range series {
		for i := 0; i < s.Len(); i++ {
			val, ok := s.Get(i)
			if ok && val != nil {
				labelSet[fmt.Sprint(val)] = true
			}
		}
	}
	
	labels := make([]string, 0, len(labelSet))
	for label := range labelSet {
		labels = append(labels, label)
	}
	
	return labels
}

func binaryPrecision(cm map[string]map[string]int) float64 {
	// Assumes binary classification with labels that can be converted to positive/negative
	// Simplified: use first label as positive class
	labels := make([]string, 0, len(cm))
	for label := range cm {
		labels = append(labels, label)
	}
	if len(labels) < 2 {
		return 0
	}
	
	posLabel := labels[0]
	tp := cm[posLabel][posLabel]
	
	fp := 0
	for _, label := range labels {
		if label != posLabel {
			fp += cm[label][posLabel]
		}
	}
	
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

func binaryRecall(cm map[string]map[string]int) float64 {
	labels := make([]string, 0, len(cm))
	for label := range cm {
		labels = append(labels, label)
	}
	if len(labels) < 2 {
		return 0
	}
	
	posLabel := labels[0]
	tp := cm[posLabel][posLabel]
	
	fn := 0
	for _, label := range labels {
		if label != posLabel {
			fn += cm[posLabel][label]
		}
	}
	
	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

func microPrecision(cm map[string]map[string]int) float64 {
	tp := 0
	fpPlusTp := 0
	
	for _, innerMap := range cm {
		for _, count := range innerMap {
			fpPlusTp += count
		}
	}
	
	for label := range cm {
		tp += cm[label][label]
	}
	
	if fpPlusTp == 0 {
		return 0
	}
	return float64(tp) / float64(fpPlusTp)
}

func microRecall(cm map[string]map[string]int) float64 {
	// For multiclass, micro precision = micro recall = micro F1 = accuracy
	return microPrecision(cm)
}

func macroPrecision(cm map[string]map[string]int) float64 {
	labels := make([]string, 0, len(cm))
	for label := range cm {
		labels = append(labels, label)
	}
	
	sum := 0.0
	for _, label := range labels {
		tp := cm[label][label]
		fp := 0
		for _, otherLabel := range labels {
			if otherLabel != label {
				fp += cm[otherLabel][label]
			}
		}
		
		if tp+fp > 0 {
			sum += float64(tp) / float64(tp+fp)
		}
	}
	
	if len(labels) == 0 {
		return 0
	}
	return sum / float64(len(labels))
}

func macroRecall(cm map[string]map[string]int) float64 {
	labels := make([]string, 0, len(cm))
	for label := range cm {
		labels = append(labels, label)
	}
	
	sum := 0.0
	for _, label := range labels {
		tp := cm[label][label]
		fn := 0
		for _, otherLabel := range labels {
			if otherLabel != label {
				fn += cm[label][otherLabel]
			}
		}
		
		if tp+fn > 0 {
			sum += float64(tp) / float64(tp+fn)
		}
	}
	
	if len(labels) == 0 {
		return 0
	}
	return sum / float64(len(labels))
}

func weightedPrecision(cm map[string]map[string]int, yTrue *seriesPkg.Series[any]) float64 {
	// Count support for each class
	support := make(map[string]int)
	for i := 0; i < yTrue.Len(); i++ {
		val, ok := yTrue.Get(i)
		if ok && val != nil {
			support[fmt.Sprint(val)]++
		}
	}
	
	labels := make([]string, 0, len(cm))
	for label := range cm {
		labels = append(labels, label)
	}
	
	weightedSum := 0.0
	totalSupport := 0
	
	for _, label := range labels {
		tp := cm[label][label]
		fp := 0
		for _, otherLabel := range labels {
			if otherLabel != label {
				fp += cm[otherLabel][label]
			}
		}
		
		if tp+fp > 0 {
			precision := float64(tp) / float64(tp+fp)
			weightedSum += precision * float64(support[label])
		}
		totalSupport += support[label]
	}
	
	if totalSupport == 0 {
		return 0
	}
	return weightedSum / float64(totalSupport)
}

func weightedRecall(cm map[string]map[string]int, yTrue *seriesPkg.Series[any]) float64 {
	support := make(map[string]int)
	for i := 0; i < yTrue.Len(); i++ {
		val, ok := yTrue.Get(i)
		if ok && val != nil {
			support[fmt.Sprint(val)]++
		}
	}
	
	labels := make([]string, 0, len(cm))
	for label := range cm {
		labels = append(labels, label)
	}
	
	weightedSum := 0.0
	totalSupport := 0
	
	for _, label := range labels {
		tp := cm[label][label]
		fn := 0
		for _, otherLabel := range labels {
			if otherLabel != label {
				fn += cm[label][otherLabel]
			}
		}
		
		if tp+fn > 0 {
			recall := float64(tp) / float64(tp+fn)
			weightedSum += recall * float64(support[label])
		}
		totalSupport += support[label]
	}
	
	if totalSupport == 0 {
		return 0
	}
	return weightedSum / float64(totalSupport)
}

func toFloat64Metrics(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	case int16:
		return float64(v)
	case int8:
		return float64(v)
	default:
		return 0
	}
}
