package linear

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
	"gonum.org/v1/gonum/mat"
)

// Ridge implements Ridge Regression with L2 regularization.
// Ridge regression adds a penalty term α * ||w||² to the loss function,
// which helps prevent overfitting by shrinking coefficients.
// Solves: β = (X^T X + α I)^-1 X^T y
type Ridge struct {
	// Alpha is the regularization strength (λ or α).
	// Larger values specify stronger regularization.
	// Alpha must be >= 0. When alpha = 0, Ridge is equivalent to LinearRegression.
	Alpha float64
	
	// FitIntercept determines whether to calculate the intercept.
	FitIntercept bool
	
	// coef stores the coefficients
	coef []float64
	
	// intercept stores the intercept term
	intercept float64
	
	// fitted indicates whether the model has been fitted
	fitted bool
	
	// featureNames stores the names of features
	featureNames []string
}

// NewRidge creates a new Ridge regression model.
func NewRidge(alpha float64, fitIntercept bool) *Ridge {
	if alpha < 0 {
		alpha = 0
	}
	return &Ridge{
		Alpha:        alpha,
		FitIntercept: fitIntercept,
		fitted:       false,
	}
}

// Fit trains the Ridge regression model.
// Solves: β = (X^T X + α I)^-1 X^T y
func (r *Ridge) Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error {
	// Extract features
	features, names, err := extractFeatures(X)
	if err != nil {
		return err
	}
	r.featureNames = names
	
	// Extract target
	target, err := extractTarget(y)
	if err != nil {
		return err
	}
	
	if len(features) != len(target) {
		return fmt.Errorf("X and y must have the same number of samples")
	}
	
	n := len(features)
	p := len(features[0])
	
	// Center data if fitting intercept
	var meanX []float64
	var meanY float64
	if r.FitIntercept {
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
	
	// Convert to matrix
	XMat := mat.NewDense(n, p, nil)
	for i, row := range features {
		XMat.SetRow(i, row)
	}
	yVec := mat.NewVecDense(n, target)
	
	// Compute X^T X
	var XTX mat.Dense
	XTX.Mul(XMat.T(), XMat)
	
	// Add regularization: X^T X + α I
	for i := 0; i < p; i++ {
		val := XTX.At(i, i)
		XTX.Set(i, i, val+r.Alpha)
	}
	
	// Invert (X^T X + α I)
	var XTXInv mat.Dense
	err = XTXInv.Inverse(&XTX)
	if err != nil {
		return fmt.Errorf("matrix inversion failed: %w", err)
	}
	
	// Compute X^T y
	var XTy mat.VecDense
	XTy.MulVec(XMat.T(), yVec)
	
	// Solve: β = (X^T X + α I)^-1 X^T y
	var beta mat.VecDense
	beta.MulVec(&XTXInv, &XTy)
	
	// Extract coefficients
	r.coef = make([]float64, p)
	for i := 0; i < p; i++ {
		r.coef[i] = beta.AtVec(i)
	}
	
	// Compute intercept if needed
	if r.FitIntercept {
		r.intercept = meanY
		for j := 0; j < p; j++ {
			r.intercept -= r.coef[j] * meanX[j]
		}
	} else {
		r.intercept = 0
	}
	
	r.fitted = true
	return nil
}

// Predict makes predictions on new data.
func (r *Ridge) Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	if !r.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	features, _, err := extractFeatures(X)
	if err != nil {
		return nil, err
	}
	
	n := len(features)
	predictions := make([]any, n)
	
	for i, row := range features {
		if len(row) != len(r.coef) {
			return nil, fmt.Errorf("feature count mismatch")
		}
		
		pred := r.intercept
		for j, x := range row {
			pred += x * r.coef[j]
		}
		predictions[i] = pred
	}
	
	return seriesPkg.New("predictions", predictions, core.DtypeFloat64), nil
}

// Coef returns the coefficients.
func (r *Ridge) Coef() []float64 {
	return r.coef
}

// Intercept returns the intercept.
func (r *Ridge) Intercept() float64 {
	return r.intercept
}

// Score returns the R² score on test data.
func (r *Ridge) Score(X *dataframe.DataFrame, y *seriesPkg.Series[any]) (float64, error) {
	yPred, err := r.Predict(X)
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
		ssRes += math.Pow(yTrue[i]-yPredVals[i], 2)
		ssTot += math.Pow(yTrue[i]-mean, 2)
	}
	
	if ssTot == 0 {
		return 0, fmt.Errorf("total sum of squares is zero")
	}
	
	return 1 - (ssRes / ssTot), nil
}
