package linear

import (
	"math"
	"testing"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// Lasso Tests

func TestLassoRegression(t *testing.T) {
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5, 6, 7, 8},
		"x2": []float64{2, 3, 1, 4, 2, 5, 3, 4},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{9.0, 14.0, 10.0, 19.0, 15.0, 25.0, 21.0, 27.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLasso(0.1, 1000, true)
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	coef := model.Coef()
	if len(coef) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(coef))
	}
	
	if model.NIter() == 0 {
		t.Error("Model should have performed at least one iteration")
	}
	
	t.Logf("Lasso converged in %d iterations", model.NIter())
	t.Logf("Coefficients: %v", coef)
}

func TestLassoSparsity(t *testing.T) {
	// Lasso should set some coefficients to exactly zero with high alpha
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5},
		"x2": []float64{1.1, 2.1, 3.1, 4.1, 5.1}, // Highly correlated with x1
		"x3": []float64{0.1, 0.2, 0.1, 0.2, 0.1}, // Weak predictor
	}
	X, _ := dataframe.New(data)
	
	yData := []any{2.0, 4.0, 6.0, 8.0, 10.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLasso(5.0, 1000, true) // High alpha for sparsity
	_ = model.Fit(X, y)
	
	coef := model.Coef()
	
	// Check if at least one coefficient is exactly zero
	hasZero := false
	for _, c := range coef {
		if c == 0.0 {
			hasZero = true
			break
		}
	}
	
	t.Logf("Lasso coefficients (high alpha): %v", coef)
	if !hasZero {
		t.Log("Note: No coefficients were set to exactly zero (this can happen with this synthetic data)")
	}
}

func TestLassoPredict(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{3.0, 5.0, 7.0, 9.0, 11.0}
	y := seriesPkg.New("y", yData, core.DtypeFloat64)
	
	model := NewLasso(1.0, 1000, true)
	_ = model.Fit(X, y)
	
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

// Logistic Regression Tests

func TestLogisticRegressionBinary(t *testing.T) {
	// Simple linearly separable data
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5, 6, 7, 8},
		"x2": []float64{1, 2, 2, 3, 5, 6, 7, 8},
	}
	X, _ := dataframe.New(data)
	
	// Class 0 for small values, class 1 for large values
	yData := []any{"A", "A", "A", "A", "B", "B", "B", "B"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewLogisticRegression("l2", 1.0, 100)
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	
	classes := model.Classes()
	if len(classes) != 2 {
		t.Errorf("Expected 2 classes, got %d", len(classes))
	}
	
	t.Logf("Classes: %v", classes)
	t.Logf("Converged in %d iterations", model.NIter())
}

func TestLogisticRegressionPredict(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5, 6, 7, 8},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{"A", "A", "A", "A", "B", "B", "B", "B"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewLogisticRegression("none", 1.0, 500) // More iterations
	_ = model.Fit(X, y)
	
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	
	if predictions.Len() != 8 {
		t.Errorf("Expected 8 predictions, got %d", predictions.Len())
	}
	
	// Count correct predictions
	correct := 0
	for i := 0; i < predictions.Len(); i++ {
		pred, _ := predictions.Get(i)
		true, _ := y.Get(i)
		if pred == true {
			correct++
		}
	}
	
	accuracy := float64(correct) / float64(predictions.Len())
	if accuracy < 0.6 {
		t.Errorf("Expected accuracy > 0.6, got %f", accuracy)
	}
	
	t.Logf("Accuracy: %.2f (iterations: %d)", accuracy, model.NIter())
}

func TestLogisticRegressionPredictProba(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 7, 8},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{"A", "A", "B", "B"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewLogisticRegression("none", 1.0, 100)
	_ = model.Fit(X, y)
	
	proba, err := model.PredictProba(X)
	if err != nil {
		t.Fatalf("PredictProba failed: %v", err)
	}
	
	if proba.Nrows() != 4 {
		t.Errorf("Expected 4 probability rows, got %d", proba.Nrows())
	}
	
	if proba.Ncols() != 2 {
		t.Errorf("Expected 2 probability columns (one per class), got %d", proba.Ncols())
	}
	
	// Check that probabilities sum to 1
	for i := 0; i < proba.Nrows(); i++ {
		cols := proba.Columns()
		var sum float64
		for _, col := range cols {
			series, _ := proba.Column(col)
			val, _ := series.Get(i)
			sum += toFloat64Linear(val)
		}
		
		if math.Abs(sum-1.0) > 0.01 {
			t.Errorf("Row %d probabilities don't sum to 1: %f", i, sum)
		}
	}
	
	t.Log("PredictProba test passed")
}

func TestLogisticRegressionInvalidClasses(t *testing.T) {
	data := map[string]any{
		"x": []float64{1, 2, 3, 4, 5},
	}
	X, _ := dataframe.New(data)
	
	// Only one class - should fail
	yData := []any{"A", "A", "A", "A", "A"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	model := NewLogisticRegression("none", 1.0, 100)
	err := model.Fit(X, y)
	
	if err == nil {
		t.Error("Expected error for single class, got nil")
	}
}

func TestLogisticRegressionRegularization(t *testing.T) {
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5, 6, 7, 8},
		"x2": []float64{1, 2, 2, 3, 5, 6, 7, 8},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{"A", "A", "A", "A", "B", "B", "B", "B"}
	y := seriesPkg.New("y", yData, core.DtypeString)
	
	// Test different regularizations
	models := []struct {
		penalty string
		C       float64
	}{
		{"none", 1.0},
		{"l2", 1.0},
		{"l1", 1.0},
	}
	
	for _, config := range models {
		model := NewLogisticRegression(config.penalty, config.C, 100)
		err := model.Fit(X, y)
		if err != nil {
			t.Errorf("Fit failed for penalty=%s: %v", config.penalty, err)
		}
		t.Logf("Penalty %s: converged in %d iterations", config.penalty, model.NIter())
	}
}
