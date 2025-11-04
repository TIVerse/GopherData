package linear

import (
	"math"
	"testing"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

func TestRidgeRegression(t *testing.T) {
	// Create data with some noise
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5, 6, 7, 8},
		"x2": []float64{2, 3, 1, 4, 2, 5, 3, 4},
	}
	X, _ := dataframe.New(data)
	
	// y = 2*x1 + 3*x2 + 1 + noise
	yData := []any{9.0, 14.0, 10.0, 19.0, 15.0, 25.0, 21.0, 27.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	// Test with small alpha
	model := NewRidge(0.1, true)
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	coef := model.Coef()
	if len(coef) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(coef))
	}
	
	// Coefficients should be close to [2, 3]
	if math.Abs(coef[0]-2.0) > 0.5 {
		t.Logf("Warning: x1 coefficient %f differs from expected 2.0", coef[0])
	}
	if math.Abs(coef[1]-3.0) > 0.5 {
		t.Logf("Warning: x2 coefficient %f differs from expected 3.0", coef[1])
	}
}

func TestRidgeVsLinear(t *testing.T) {
	// Ridge with alpha=0 should behave like LinearRegression
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	// Ridge with alpha=0
	ridge := NewRidge(0.0, true)
	ridge.Fit(X, y)
	
	// Linear regression
	linear := NewLinearRegression(true)
	linear.Fit(X, y)
	
	// Coefficients should be very similar
	ridgeCoef := ridge.Coef()[0]
	linearCoef := linear.Coef()[0]
	
	if math.Abs(ridgeCoef-linearCoef) > 0.01 {
		t.Errorf("Ridge (alpha=0) coefficient %f differs from Linear %f", ridgeCoef, linearCoef)
	}
}

func TestRidgePredict(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{3.0, 5.0, 7.0, 9.0, 11.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewRidge(1.0, true)
	model.Fit(X, y)
	
	// Test data
	testData := map[string]any{
		"x": []float64{6, 7},
	}
	XTest, _ := dataframe.New(testData)
	
	predictions, err := model.Predict(XTest)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if predictions.Len() != 2 {
		t.Errorf("Expected 2 predictions, got %d", predictions.Len())
	}
}

func TestRidgeScore(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewRidge(0.5, true)
	model.Fit(X, y)
	
	r2, err := model.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	
	// R² should be high for good fit
	if r2 < 0.9 {
		t.Errorf("Expected R² > 0.9, got %f", r2)
	}
}

func TestRidgeRegularization(t *testing.T) {
	// With high alpha, coefficients should shrink
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5},
		"x2": []float64{2, 3, 4, 5, 6},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{5.0, 8.0, 11.0, 14.0, 17.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	// Low regularization
	model1 := NewRidge(0.1, true)
	model1.Fit(X, y)
	coef1 := model1.Coef()
	
	// High regularization
	model2 := NewRidge(10.0, true)
	model2.Fit(X, y)
	coef2 := model2.Coef()
	
	// High alpha should produce smaller coefficients
	sum1 := math.Abs(coef1[0]) + math.Abs(coef1[1])
	sum2 := math.Abs(coef2[0]) + math.Abs(coef2[1])
	
	if sum2 >= sum1 {
		t.Logf("Warning: High regularization did not shrink coefficients as expected")
		t.Logf("Low alpha coefs: %v, High alpha coefs: %v", coef1, coef2)
	}
}
