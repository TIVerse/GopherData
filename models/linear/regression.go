// Package linear provides linear regression models.
package linear

import (
	"fmt"
	//"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
	"gonum.org/v1/gonum/mat"
)

// LinearRegression implements Ordinary Least Squares (OLS) regression.
// Fits a linear model with coefficients w = (w_1, ..., w_p) to minimize
// the residual sum of squares between the observed targets and the targets
// predicted by the linear approximation.
type LinearRegression struct {
	// FitIntercept determines whether to calculate the intercept.
	// If false, the data is expected to be centered.
	FitIntercept bool

	// coef stores the coefficients of the linear model
	coef []float64

	// intercept stores the intercept term
	intercept float64

	// fitted indicates whether the model has been fitted
	fitted bool

	// featureNames stores the names of features
	featureNames []string
}

// NewLinearRegression creates a new LinearRegression model.
func NewLinearRegression(fitIntercept bool) *LinearRegression {
	return &LinearRegression{
		FitIntercept: fitIntercept,
		fitted:       false,
	}
}

// Fit trains the linear regression model on data X with target y.
// Solves the normal equation: β = (X^T X)^-1 X^T y
func (lr *LinearRegression) Fit(X *dataframe.DataFrame, y *seriesPkg.Series[any]) error {
	// Extract numeric features
	features, names, err := extractFeatures(X)
	if err != nil {
		return err
	}
	lr.featureNames = names

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

	// Add intercept column if needed
	if lr.FitIntercept {
		for i := range features {
			features[i] = append([]float64{1.0}, features[i]...)
		}
		p++
	}

	// Convert to matrix
	XMat := mat.NewDense(n, p, nil)
	for i, row := range features {
		XMat.SetRow(i, row)
	}

	yVec := mat.NewVecDense(n, target)

	// Solve normal equation: (X^T X)^-1 X^T y
	var XTX mat.Dense
	XTX.Mul(XMat.T(), XMat)

	var XTXInv mat.Dense
	err = XTXInv.Inverse(&XTX)
	if err != nil {
		return fmt.Errorf("matrix is singular, cannot solve normal equation: %w", err)
	}

	var XTy mat.VecDense
	XTy.MulVec(XMat.T(), yVec)

	var beta mat.VecDense
	beta.MulVec(&XTXInv, &XTy)

	// Extract coefficients
	if lr.FitIntercept {
		lr.intercept = beta.AtVec(0)
		lr.coef = make([]float64, p-1)
		for i := 1; i < p; i++ {
			lr.coef[i-1] = beta.AtVec(i)
		}
	} else {
		lr.intercept = 0
		lr.coef = make([]float64, p)
		for i := 0; i < p; i++ {
			lr.coef[i] = beta.AtVec(i)
		}
	}

	lr.fitted = true
	return nil
}

// Predict makes predictions on new data X.
// Returns y_pred = X * coef + intercept
func (lr *LinearRegression) Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	if !lr.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}

	features, _, err := extractFeatures(X)
	if err != nil {
		return nil, err
	}

	n := len(features)
	predictions := make([]any, n)

	for i, row := range features {
		if len(row) != len(lr.coef) {
			return nil, fmt.Errorf("feature count mismatch: expected %d, got %d", len(lr.coef), len(row))
		}

		pred := lr.intercept
		for j, x := range row {
			pred += x * lr.coef[j]
		}
		predictions[i] = pred
	}

	return seriesPkg.New("predictions", predictions, core.DtypeFloat64), nil
}

// Coef returns the coefficients of the model.
func (lr *LinearRegression) Coef() []float64 {
	return lr.coef
}

// Intercept returns the intercept of the model.
func (lr *LinearRegression) Intercept() float64 {
	return lr.intercept
}

// Score returns the R² score of the model on test data.
func (lr *LinearRegression) Score(X *dataframe.DataFrame, y *seriesPkg.Series[any]) (float64, error) {
	yPred, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	// Calculate R²
	yTrue, err := extractTarget(y)
	if err != nil {
		return 0, err
	}

	yPredVals := make([]float64, yPred.Len())
	for i := 0; i < yPred.Len(); i++ {
		val, _ := yPred.Get(i)
		yPredVals[i] = toFloat64Linear(val)
	}

	// Calculate mean of true values
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))

	// Calculate SS_res and SS_tot
	var ssRes, ssTot float64
	for i := range yTrue {
		diffRes := yTrue[i] - yPredVals[i]
		ssRes += diffRes * diffRes
		diffTot := yTrue[i] - mean
		ssTot += diffTot * diffTot
	}

	if ssTot == 0 {
		return 0, fmt.Errorf("total sum of squares is zero")
	}

	return 1 - (ssRes / ssTot), nil
}

// Helper functions

func extractFeatures(X *dataframe.DataFrame) ([][]float64, []string, error) {
	n := X.Nrows()
	cols := X.Columns()

	// Filter numeric columns
	numericCols := make([]string, 0)
	for _, col := range cols {
		series, err := X.Column(col)
		if err != nil {
			continue
		}
		if isNumeric(series.Dtype()) {
			numericCols = append(numericCols, col)
		}
	}

	if len(numericCols) == 0 {
		return nil, nil, fmt.Errorf("no numeric columns found")
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
				features[i][j] = toFloat64Linear(val)
			}
		}
	}

	return features, numericCols, nil
}

func extractTarget(y *seriesPkg.Series[any]) ([]float64, error) {
	n := y.Len()
	target := make([]float64, n)

	for i := 0; i < n; i++ {
		val, ok := y.Get(i)
		if !ok || val == nil {
			return nil, fmt.Errorf("target contains null values at index %d", i)
		}
		target[i] = toFloat64Linear(val)
	}

	return target, nil
}

func isNumeric(dtype core.Dtype) bool {
	switch dtype {
	case core.DtypeFloat64, core.DtypeInt64:
		return true
	}
	return false
}

func toFloat64Linear(val any) float64 {
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
