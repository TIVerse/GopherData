package linear

import (
	"math"
	"testing"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

func TestLinearRegressionSimple(t *testing.T) {
	// Create simple linear data: y = 2x + 1
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{3.0, 5.0, 7.0, 9.0, 11.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLinearRegression(true)
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// Check coefficients
	coef := model.Coef()
	if len(coef) != 1 {
		t.Errorf("Expected 1 coefficient, got %d", len(coef))
	}
	
	if math.Abs(coef[0]-2.0) > 0.01 {
		t.Errorf("Expected coefficient ~2.0, got %f", coef[0])
	}
	
	if math.Abs(model.Intercept()-1.0) > 0.01 {
		t.Errorf("Expected intercept ~1.0, got %f", model.Intercept())
	}
}

func TestLinearRegressionPredict(t *testing.T) {
	// Training data
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{3.0, 5.0, 7.0, 9.0, 11.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLinearRegression(true)
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	// Test data
	testData := map[string]any{
		"x": []float64{6, 7, 8},
	}
	XTest, _ := dataframe.New(testData)
	
	predictions, err := model.Predict(XTest)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if predictions.Len() != 3 {
		t.Errorf("Expected 3 predictions, got %d", predictions.Len())
	}
	
	// Check predictions (should be approximately 13, 15, 17)
	expected := []float64{13.0, 15.0, 17.0}
	for i, exp := range expected {
		val, _ := predictions.Get(i)
		pred := toFloat64Linear(val)
		if math.Abs(pred-exp) > 0.1 {
			t.Errorf("Prediction %d: expected ~%f, got %f", i, exp, pred)
		}
	}
}

func TestLinearRegressionMultipleFeatures(t *testing.T) {
	// y = 2*x1 + 3*x2 + 1
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5},
		"x2": []float64{1, 1, 2, 2, 3},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{6.0, 8.0, 13.0, 15.0, 20.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLinearRegression(true)
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	coef := model.Coef()
	if len(coef) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(coef))
	}
	
	// Coefficients should sum to approximately 5 (2+3)
	// Note: Order may vary depending on DataFrame column ordering
	coefSum := coef[0] + coef[1]
	if math.Abs(coefSum-5.0) > 0.5 {
		t.Errorf("Expected coefficient sum ~5.0, got %f (coefs: %v)", coefSum, coef)
	}
	
	t.Logf("Coefficients: %v (sum: %.2f)", coef, coefSum)
}

func TestLinearRegressionScore(t *testing.T) {
	// Perfect linear relationship
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLinearRegression(true)
	if err := model.Fit(X, y); err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	r2, err := model.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	
	// R² should be very close to 1.0 for perfect fit
	if r2 < 0.99 {
		t.Errorf("Expected R² ~1.0, got %f", r2)
	}
}

func TestLinearRegressionNoIntercept(t *testing.T) {
	// y = 2x (no intercept)
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLinearRegression(false)
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	if model.Intercept() != 0 {
		t.Errorf("Expected intercept 0, got %f", model.Intercept())
	}
	
	coef := model.Coef()
	if math.Abs(coef[0]-2.0) > 0.01 {
		t.Errorf("Expected coefficient ~2.0, got %f", coef[0])
	}
}
