package linear

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// LogisticRegression implements logistic regression for binary classification.
// Uses gradient descent with logistic loss function.
// Supports L1, L2, or no regularization.
type LogisticRegression struct {
	// Penalty specifies the regularization type: "l1", "l2", or "none"
	Penalty string
	
	// C is the inverse of regularization strength (smaller values = stronger regularization)
	C float64
	
	// MaxIter is the maximum number of iterations for gradient descent
	MaxIter int
	
	// Tol is the tolerance for convergence
	Tol float64
	
	// FitIntercept determines whether to calculate the intercept
	FitIntercept bool
	
	// LearningRate for gradient descent
	LearningRate float64
	
	// coef stores the coefficients
	coef []float64
	
	// intercept stores the intercept term
	intercept float64
	
	// fitted indicates whether the model has been fitted
	fitted bool
	
	// classes stores the unique class labels
	classes []string
	
	// nIter stores the actual number of iterations performed
	nIter int
	
	// featureNames stores the names of features
	featureNames []string
}

// NewLogisticRegression creates a new logistic regression model.
func NewLogisticRegression(penalty string, C float64, maxIter int) *LogisticRegression {
	if C <= 0 {
		C = 1.0
	}
	if maxIter <= 0 {
		maxIter = 100
	}
	
	return &LogisticRegression{
		Penalty:      penalty,
		C:            C,
		MaxIter:      maxIter,
		Tol:          1e-4,
		FitIntercept: true,
		LearningRate: 0.1, // Increased for faster convergence
		fitted:       false,
	}
}

// Fit trains the logistic regression model using gradient descent.
func (lr *LogisticRegression) Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error {
	// Extract features
	features, names, err := extractFeatures(X)
	if err != nil {
		return err
	}
	lr.featureNames = names
	
	// Extract and encode target
	labels := make([]string, y.Len())
	for i := 0; i < y.Len(); i++ {
		val, ok := y.Get(i)
		if !ok || val == nil {
			return fmt.Errorf("target contains null at index %d", i)
		}
		labels[i] = fmt.Sprint(val)
	}
	
	// Get unique classes
	classSet := make(map[string]bool)
	for _, label := range labels {
		classSet[label] = true
	}
	
	lr.classes = make([]string, 0, len(classSet))
	for class := range classSet {
		lr.classes = append(lr.classes, class)
	}
	
	if len(lr.classes) != 2 {
		return fmt.Errorf("logistic regression requires exactly 2 classes, got %d", len(lr.classes))
	}
	
	// Encode labels as 0/1
	target := make([]float64, len(labels))
	for i, label := range labels {
		if label == lr.classes[1] {
			target[i] = 1.0
		} else {
			target[i] = 0.0
		}
	}
	
	if len(features) != len(target) {
		return fmt.Errorf("X and y must have the same number of samples")
	}
	
	n := len(features)
	p := len(features[0])
	
	// Initialize coefficients
	lr.coef = make([]float64, p)
	lr.intercept = 0
	
	// Gradient descent
	alpha := 1.0 / (lr.C * float64(n)) // Regularization strength
	
	for iter := 0; iter < lr.MaxIter; iter++ {
		// Compute predictions
		predictions := make([]float64, n)
		for i := 0; i < n; i++ {
			z := lr.intercept
			for j := 0; j < p; j++ {
				z += lr.coef[j] * features[i][j]
			}
			predictions[i] = sigmoid(z)
		}
		
		// Compute gradients
		gradCoef := make([]float64, p)
		gradIntercept := 0.0
		
		for i := 0; i < n; i++ {
			error := predictions[i] - target[i]
			gradIntercept += error
			for j := 0; j < p; j++ {
				gradCoef[j] += error * features[i][j]
			}
		}
		
		// Add regularization gradient
		if lr.Penalty == "l2" {
			for j := 0; j < p; j++ {
				gradCoef[j] += alpha * lr.coef[j]
			}
		} else if lr.Penalty == "l1" {
			for j := 0; j < p; j++ {
				if lr.coef[j] > 0 {
					gradCoef[j] += alpha
				} else if lr.coef[j] < 0 {
					gradCoef[j] -= alpha
				}
			}
		}
		
		// Update coefficients
		maxChange := 0.0
		for j := 0; j < p; j++ {
			change := lr.LearningRate * gradCoef[j] / float64(n)
			lr.coef[j] -= change
			if math.Abs(change) > maxChange {
				maxChange = math.Abs(change)
			}
		}
		
		if lr.FitIntercept {
			lr.intercept -= lr.LearningRate * gradIntercept / float64(n)
		}
		
		lr.nIter = iter + 1
		
		// Check convergence
		if maxChange < lr.Tol {
			break
		}
	}
	
	lr.fitted = true
	return nil
}

// Predict predicts class labels for samples in X.
func (lr *LogisticRegression) Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	if !lr.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	proba, err := lr.PredictProba(X)
	if err != nil {
		return nil, err
	}
	
	// Get probability of class 1
	probaCol, err := proba.Column(lr.classes[1])
	if err != nil {
		return nil, err
	}
	
	predictions := make([]any, probaCol.Len())
	for i := 0; i < probaCol.Len(); i++ {
		val, _ := probaCol.Get(i)
		prob := toFloat64Linear(val)
		
		if prob >= 0.5 {
			predictions[i] = lr.classes[1]
		} else {
			predictions[i] = lr.classes[0]
		}
	}
	
	return seriesPkg.New("predictions", predictions, core.DtypeString), nil
}

// PredictProba returns probability estimates for each class.
func (lr *LogisticRegression) PredictProba(X *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !lr.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	features, _, err := extractFeatures(X)
	if err != nil {
		return nil, err
	}
	
	n := len(features)
	proba0 := make([]any, n)
	proba1 := make([]any, n)
	
	for i, row := range features {
		if len(row) != len(lr.coef) {
			return nil, fmt.Errorf("feature count mismatch")
		}
		
		z := lr.intercept
		for j, x := range row {
			z += x * lr.coef[j]
		}
		
		p1 := sigmoid(z)
		p0 := 1 - p1
		
		proba0[i] = p0
		proba1[i] = p1
	}
	
	probaData := map[string]any{
		lr.classes[0]: proba0,
		lr.classes[1]: proba1,
	}
	
	return dataframe.New(probaData)
}

// Coef returns the coefficients.
func (lr *LogisticRegression) Coef() []float64 {
	return lr.coef
}

// Intercept returns the intercept.
func (lr *LogisticRegression) Intercept() float64 {
	return lr.intercept
}

// Classes returns the class labels.
func (lr *LogisticRegression) Classes() []string {
	return lr.classes
}

// NIter returns the number of iterations performed.
func (lr *LogisticRegression) NIter() int {
	return lr.nIter
}

// sigmoid computes the logistic sigmoid function: 1 / (1 + exp(-z))
func sigmoid(z float64) float64 {
	if z > 20 {
		return 1.0 // Avoid overflow
	} else if z < -20 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-z))
}
