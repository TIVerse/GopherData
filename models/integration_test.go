package models

import (
	"testing"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/models/linear"
	seriesPkg "github.com/TIVerse/GopherData/series"
	"github.com/TIVerse/GopherData/stats"
)

// TestEndToEndMLPipeline demonstrates a complete ML workflow
func TestEndToEndMLPipeline(t *testing.T) {
	// Create synthetic dataset
	// y = 3*x1 + 2*x2 + 1 + noise
	data := map[string]any{
		"x1": []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		"x2": []float64{2, 3, 1, 4, 2, 5, 3, 4, 2, 6},
	}
	X, _ := dataframe.New(data)
	
	yData := []any{8.0, 13.0, 12.0, 19.0, 17.0, 24.0, 23.0, 29.0, 26.0, 35.0}
	y := seriesPkg.New("price", yData, core.DtypeFloat64)
	
	// Descriptive statistics
	x1Vals := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	summary := stats.Describe(x1Vals)
	
	if summary.Count != 10 {
		t.Errorf("Expected count 10, got %d", summary.Count)
	}
	
	// Train/test split (80/20)
	split, err := TrainTestSplitFunc(X, y, 0.2, true, "", 42)
	if err != nil {
		t.Fatalf("TrainTestSplit failed: %v", err)
	}
	
	if split.XTrain.Nrows() != 8 {
		t.Errorf("Expected 8 training samples, got %d", split.XTrain.Nrows())
	}
	if split.XTest.Nrows() != 2 {
		t.Errorf("Expected 2 test samples, got %d", split.XTest.Nrows())
	}
	
	// Train linear regression model
	model := linear.NewLinearRegression(true)
	err = model.Fit(split.XTrain, split.YTrain)
	if err != nil {
		t.Fatalf("Model fit failed: %v", err)
	}
	
	// Check coefficients (should be approximately [3, 2])
	coef := model.Coef()
	if len(coef) != 2 {
		t.Errorf("Expected 2 coefficients, got %d", len(coef))
	}
	
	// Make predictions
	yPred, err := model.Predict(split.XTest)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}
	
	if yPred.Len() != split.YTest.Len() {
		t.Errorf("Prediction length mismatch")
	}
	
	// Evaluate model
	r2, err := model.Score(split.XTest, split.YTest)
	if err != nil {
		t.Fatalf("Score calculation failed: %v", err)
	}
	
	// R² should be high for this synthetic data
	if r2 < 0.8 {
		t.Errorf("Expected R² > 0.8, got %f", r2)
	}
	
	// Calculate metrics
	mse := MSE(split.YTest, yPred)
	rmse := RMSE(split.YTest, yPred)
	mae := MAE(split.YTest, yPred)
	
	if mse < 0 {
		t.Error("MSE should be non-negative")
	}
	if rmse < 0 {
		t.Error("RMSE should be non-negative")
	}
	if mae < 0 {
		t.Error("MAE should be non-negative")
	}
	
	t.Logf("Model Performance:")
	t.Logf("  R²: %.4f", r2)
	t.Logf("  MSE: %.4f", mse)
	t.Logf("  RMSE: %.4f", rmse)
	t.Logf("  MAE: %.4f", mae)
	t.Logf("  Coefficients: %v", coef)
	t.Logf("  Intercept: %.4f", model.Intercept())
}

// TestCorrelationAnalysis tests correlation functions
func TestCorrelationAnalysis(t *testing.T) {
	// Create correlated data
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	y := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20} // Perfect correlation: y = 2x
	
	// Test Pearson correlation
	corr, err := stats.Pearson(x, y)
	if err != nil {
		t.Fatalf("Pearson correlation failed: %v", err)
	}
	
	// Should be very close to 1.0 for perfect linear relationship
	if corr < 0.99 {
		t.Errorf("Expected Pearson correlation ~1.0, got %f", corr)
	}
	
	// Test Spearman correlation
	spearman, err := stats.Spearman(x, y)
	if err != nil {
		t.Fatalf("Spearman correlation failed: %v", err)
	}
	
	if spearman < 0.99 {
		t.Errorf("Expected Spearman correlation ~1.0, got %f", spearman)
	}
	
	// Test with DataFrame
	data := map[string]any{
		"feature1": x,
		"feature2": y,
	}
	df, _ := dataframe.New(data)
	
	corrMatrix, err := stats.CorrMatrix(df, "pearson")
	if err != nil {
		t.Fatalf("CorrMatrix failed: %v", err)
	}
	
	if corrMatrix.Nrows() != 2 || corrMatrix.Ncols() != 2 {
		t.Errorf("Expected 2x2 correlation matrix, got %dx%d", corrMatrix.Nrows(), corrMatrix.Ncols())
	}
	
	t.Logf("Pearson correlation: %.4f", corr)
	t.Logf("Spearman correlation: %.4f", spearman)
}

// TestMetricsComprehensive tests all metric functions
func TestMetricsComprehensive(t *testing.T) {
	// Classification metrics
	yTrueClass := seriesPkg.New("true", []any{"A", "B", "A", "B", "A", "B", "A", "A"}, core.DtypeString)
	yPredClass := seriesPkg.New("pred", []any{"A", "B", "A", "A", "A", "B", "A", "A"}, core.DtypeString)
	
	acc := Accuracy(yTrueClass, yPredClass)
	prec := Precision(yTrueClass, yPredClass, "macro")
	rec := Recall(yTrueClass, yPredClass, "macro")
	f1 := F1Score(yTrueClass, yPredClass, "macro")
	
	if acc <= 0 || acc > 1 {
		t.Errorf("Accuracy should be in [0, 1], got %f", acc)
	}
	
	t.Logf("Classification Metrics:")
	t.Logf("  Accuracy: %.4f", acc)
	t.Logf("  Precision (macro): %.4f", prec)
	t.Logf("  Recall (macro): %.4f", rec)
	t.Logf("  F1 Score (macro): %.4f", f1)
	
	// Regression metrics
	yTrueReg := seriesPkg.New("true", []any{1.0, 2.0, 3.0, 4.0, 5.0}, core.DtypeFloat64)
	yPredReg := seriesPkg.New("pred", []any{1.1, 2.1, 2.9, 4.2, 4.8}, core.DtypeFloat64)
	
	mse := MSE(yTrueReg, yPredReg)
	rmse := RMSE(yTrueReg, yPredReg)
	mae := MAE(yTrueReg, yPredReg)
	r2 := R2Score(yTrueReg, yPredReg)
	
	t.Logf("Regression Metrics:")
	t.Logf("  MSE: %.4f", mse)
	t.Logf("  RMSE: %.4f", rmse)
	t.Logf("  MAE: %.4f", mae)
	t.Logf("  R²: %.4f", r2)
}
