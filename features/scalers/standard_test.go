package scalers

import (
	"math"
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
)

func TestStandardScaler(t *testing.T) {
	t.Run("BasicScaling", func(t *testing.T) {
		data := map[string]any{
			"col1": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			"col2": []float64{10.0, 20.0, 30.0, 40.0, 50.0},
		}
		df, err := dataframe.New(data)
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		scaler := NewStandardScaler([]string{"col1", "col2"})
		
		// Fit
		if err := scaler.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Check fitted state
		if !scaler.IsFitted() {
			t.Error("Scaler should be fitted")
		}

		// Check means
		means := scaler.GetMeans()
		if !almostEqual(means["col1"], 3.0, 0.01) {
			t.Errorf("Expected mean 3.0 for col1, got %f", means["col1"])
		}
		if !almostEqual(means["col2"], 30.0, 0.01) {
			t.Errorf("Expected mean 30.0 for col2, got %f", means["col2"])
		}

		// Check stds
		stds := scaler.GetStds()
		expectedStd := math.Sqrt(2.5) // Sample std of [1,2,3,4,5]
		if !almostEqual(stds["col1"], expectedStd, 0.01) {
			t.Errorf("Expected std %f for col1, got %f", expectedStd, stds["col1"])
		}

		// Transform
		scaled, err := scaler.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if scaled.Nrows() != df.Nrows() {
			t.Errorf("Expected %d rows, got %d", df.Nrows(), scaled.Nrows())
		}
	})

	t.Run("WithNulls", func(t *testing.T) {
		data := map[string]any{
			"col": []any{1.0, nil, 3.0, 4.0, 5.0},
		}
		df, _ := dataframe.New(data)

		scaler := NewStandardScaler([]string{"col"})
		if err := scaler.Fit(df); err != nil {
			t.Fatalf("Fit with nulls failed: %v", err)
		}

		// Mean should be computed from non-null values only
		means := scaler.GetMeans()
		expectedMean := (1.0 + 3.0 + 4.0 + 5.0) / 4.0
		if !almostEqual(means["col"], expectedMean, 0.01) {
			t.Errorf("Expected mean %f, got %f", expectedMean, means["col"])
		}
	})

	t.Run("WithMeanFalse", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := dataframe.New(data)

		scaler := NewStandardScaler([]string{"col"})
		scaler.WithMean = false
		
		if err := scaler.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		means := scaler.GetMeans()
		if len(means) != 0 {
			t.Error("Should not compute means when WithMean=false")
		}
	})

	t.Run("ZeroVariance", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{5.0, 5.0, 5.0, 5.0, 5.0},
		}
		df, _ := dataframe.New(data)

		scaler := NewStandardScaler([]string{"col"})
		if err := scaler.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		stds := scaler.GetStds()
		if stds["col"] != 0 {
			t.Errorf("Expected std 0 for constant column, got %f", stds["col"])
		}

		// Transform should skip zero-variance columns
		scaled, err := scaler.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if scaled.Nrows() != df.Nrows() {
			t.Error("Transform should not fail on zero-variance columns")
		}
	})

	t.Run("NotFittedError", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		scaler := NewStandardScaler([]string{"col"})
		
		// Transform without fit should error
		_, err := scaler.Transform(df)
		if err == nil {
			t.Error("Transform should fail when not fitted")
		}
	})

	t.Run("EmptyColumns", func(t *testing.T) {
		data := map[string]any{
			"text": []string{"a", "b", "c"},
		}
		df, _ := dataframe.New(data)

		scaler := NewStandardScaler(nil) // Auto-detect numeric
		err := scaler.Fit(df)
		if err == nil {
			t.Error("Should error when no numeric columns found")
		}
	})
}

func TestStandardScalerFitTransform(t *testing.T) {
	data := map[string]any{
		"col": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
	}
	df, _ := dataframe.New(data)

	scaler := NewStandardScaler([]string{"col"})
	scaled, err := scaler.FitTransform(df)
	
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if !scaler.IsFitted() {
		t.Error("Scaler should be fitted after FitTransform")
	}

	if scaled.Nrows() != df.Nrows() {
		t.Errorf("Expected %d rows, got %d", df.Nrows(), scaled.Nrows())
	}
}

// Helper function for floating point comparison
func almostEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}
