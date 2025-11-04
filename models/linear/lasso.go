package linear

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// Lasso implements Lasso Regression with L1 regularization.
// Lasso adds a penalty term α * ||w||₁ to the loss function,
// which encourages sparsity (some coefficients become exactly zero).
// Uses coordinate descent algorithm for optimization.
type Lasso struct {
	// Alpha is the regularization strength.
	// Larger values specify stronger regularization.
	Alpha float64
	
	// MaxIter is the maximum number of iterations.
	MaxIter int
	
	// Tol is the tolerance for convergence.
	Tol float64
	
	// FitIntercept determines whether to calculate the intercept.
	FitIntercept bool
	
	// coef stores the coefficients
	coef []float64
	
	// intercept stores the intercept term
	intercept float64
	
	// fitted indicates whether the model has been fitted
	fitted bool
	
	// nIter stores the actual number of iterations performed
	nIter int
	
	// featureNames stores the names of features
	featureNames []string
}

// NewLasso creates a new Lasso regression model.
func NewLasso(alpha float64, maxIter int, fitIntercept bool) *Lasso {
	if alpha < 0 {
		alpha = 0
	}
	if maxIter <= 0 {
		maxIter = 1000
	}
	return &Lasso{
		Alpha:        alpha,
		MaxIter:      maxIter,
		Tol:          1e-4,
		FitIntercept: fitIntercept,
		fitted:       false,
	}
}

// Fit trains the Lasso regression model using coordinate descent.
func (l *Lasso) Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error {
	// Extract features
	features, names, err := extractFeatures(X)
	if err != nil {
		return err
	}
	l.featureNames = names
	
	// Extract target
	target, err := extractTarget(y)
	if err != nil {
		return err
	}
	
	if len(features) != len(target) {
		return fmt.Errorf("x and y must have the same number of samples")
	}
	
	n := len(features)
	p := len(features[0])
	
	// Center data if fitting intercept
	var meanX []float64
	var meanY float64
	if l.FitIntercept {
		meanX = make([]float64, p)
		for j := 0; j < p; j++ {
			sum := 0.0
			for i := 0; i < n; i++ {
				sum += features[i][j]
			}
			meanX[j] = sum / float64(n)
		}
		
		for i := 0; i < n; i++ {
			meanY += target[i]
		}
		meanY /= float64(n)
		
		// Center the data
		for i := 0; i < n; i++ {
			for j := 0; j < p; j++ {
				features[i][j] -= meanX[j]
			}
			target[i] -= meanY
		}
	}
	
	// Normalize features (important for Lasso)
	stdX := make([]float64, p)
	for j := 0; j < p; j++ {
		var sumSq float64
		for i := 0; i < n; i++ {
			sumSq += features[i][j] * features[i][j]
		}
		stdX[j] = math.Sqrt(sumSq / float64(n))
		if stdX[j] > 1e-10 {
			for i := 0; i < n; i++ {
				features[i][j] /= stdX[j]
			}
		}
	}
	
	// Initialize coefficients
	l.coef = make([]float64, p)
	
	// Coordinate descent
	for iter := 0; iter < l.MaxIter; iter++ {
		maxChange := 0.0
		
		for j := 0; j < p; j++ {
			oldCoef := l.coef[j]
			
			// Compute residual without feature j
			var rho float64
			for i := 0; i < n; i++ {
				residual := target[i]
				for k := 0; k < p; k++ {
					if k != j {
						residual -= features[i][k] * l.coef[k]
					}
				}
				rho += features[i][j] * residual
			}
			
			// Soft thresholding
			if rho < -l.Alpha {
				l.coef[j] = (rho + l.Alpha) / float64(n)
			} else if rho > l.Alpha {
				l.coef[j] = (rho - l.Alpha) / float64(n)
			} else {
				l.coef[j] = 0
			}
			
			// Track convergence
			change := math.Abs(l.coef[j] - oldCoef)
			if change > maxChange {
				maxChange = change
			}
		}
		
		l.nIter = iter + 1
		
		// Check convergence
		if maxChange < l.Tol {
			break
		}
	}
	
	// Rescale coefficients
	for j := 0; j < p; j++ {
		if stdX[j] > 1e-10 {
			l.coef[j] /= stdX[j]
		}
	}
	
	// Compute intercept if needed
	if l.FitIntercept {
		l.intercept = meanY
		for j := 0; j < p; j++ {
			l.intercept -= l.coef[j] * meanX[j]
		}
	} else {
		l.intercept = 0
	}
	
	l.fitted = true
	return nil
}

// Predict makes predictions on new data.
func (l *Lasso) Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	if !l.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	features, _, err := extractFeatures(X)
	if err != nil {
		return nil, err
	}
	
	n := len(features)
	predictions := make([]any, n)
	
	for i, row := range features {
		if len(row) != len(l.coef) {
			return nil, fmt.Errorf("feature count mismatch")
		}
		
		pred := l.intercept
		for j, x := range row {
			pred += x * l.coef[j]
		}
		predictions[i] = pred
	}
	
	return seriesPkg.New("predictions", predictions, core.DtypeFloat64), nil
}

// Coef returns the coefficients.
func (l *Lasso) Coef() []float64 {
	return l.coef
}

// Intercept returns the intercept.
func (l *Lasso) Intercept() float64 {
	return l.intercept
}

// NIter returns the number of iterations performed.
func (l *Lasso) NIter() int {
	return l.nIter
}

// Score returns the R² score on test data.
func (l *Lasso) Score(X *dataframe.DataFrame, y *seriesPkg.Series[any]) (float64, error) {
	yPred, err := l.Predict(X)
	if err != nil {
		return 0, err
	}
	
	yTrue, err := extractTarget(y)
	if err != nil {
		return 0, err
	}
	
	yPredVals := make([]float64, yPred.Len())
	for i := 0; i < yPred.Len(); i++ {
		val, _ := yPred.Get(i)
		yPredVals[i] = toFloat64Linear(val)
	}
	
	// Calculate R²
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))
	
	var ssRes, ssTot float64
	for i := range yTrue {
		diff := yTrue[i] - yPredVals[i]
		ssRes += diff * diff
		dev := yTrue[i] - mean
		ssTot += dev * dev
	}
	
	if ssTot == 0 {
		return 0, fmt.Errorf("total sum of squares is zero")
	}
	
	return 1 - (ssRes / ssTot), nil
}
