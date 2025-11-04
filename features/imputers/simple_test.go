package imputers

import (
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
)

func TestSimpleImputer(t *testing.T) {
	t.Run("MeanStrategy", func(t *testing.T) {
		data := map[string]any{
			"col": []any{1.0, nil, 3.0, nil, 5.0},
		}
		df, err := dataframe.New(data)
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		imputer := NewSimpleImputer([]string{"col"}, "mean")
		
		// Fit
		if err := imputer.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		if !imputer.IsFitted() {
			t.Error("Imputer should be fitted")
		}

		// Check computed mean: (1+3+5)/3 = 3.0
		stats := imputer.GetStats()
		expectedMean := 3.0
		if val, ok := stats["col"]; ok {
			if meanVal, isFloat := val.(float64); isFloat {
				if meanVal != expectedMean {
					t.Errorf("Expected mean %f, got %f", expectedMean, meanVal)
				}
			} else {
				t.Errorf("Expected float64, got %T", val)
			}
		} else {
			t.Error("Expected stats to contain 'col'")
		}

		// Transform
		imputed, err := imputer.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if imputed.Nrows() != df.Nrows() {
			t.Errorf("Expected %d rows, got %d", df.Nrows(), imputed.Nrows())
		}

		// Verify no nulls remain (this would need DataFrame.IsNA() method)
		// For now, just check that transform succeeded
	})

	t.Run("MedianStrategy", func(t *testing.T) {
		data := map[string]any{
			"col": []any{1.0, nil, 3.0, 5.0, 7.0},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer([]string{"col"}, "median")
		
		if err := imputer.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Median of [1,3,5,7] = 4.0
		stats := imputer.GetStats()
		expectedMedian := 4.0
		if val, ok := stats["col"]; ok {
			if medianVal, isFloat := val.(float64); isFloat {
				if medianVal != expectedMedian {
					t.Errorf("Expected median %f, got %f", expectedMedian, medianVal)
				}
			} else {
				t.Errorf("Expected float64, got %T", val)
			}
		} else {
			t.Error("Expected stats to contain 'col'")
		}
	})

	t.Run("ConstantStrategy", func(t *testing.T) {
		data := map[string]any{
			"col": []any{1.0, nil, 3.0},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer([]string{"col"}, "constant")
		imputer.FillValue = 999.0
		
		if err := imputer.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		stats := imputer.GetStats()
		if stats["col"].(float64) != 999.0 {
			t.Errorf("Expected fill value 999.0, got %f", stats["col"])
		}

		imputed, err := imputer.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if imputed.Nrows() != df.Nrows() {
			t.Error("Should preserve row count")
		}
	})

	t.Run("MostFrequentStrategy", func(t *testing.T) {
		data := map[string]any{
			"col": []any{"A", nil, "B", "A", "A"},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer([]string{"col"}, "most_frequent")
		
		if err := imputer.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Most frequent is "A" (appears 3 times)
		stats := imputer.GetStats()
		if stats["col"] != "A" {
			t.Errorf("Expected mode 'A', got %v", stats["col"])
		}
	})

	t.Run("NoNullsColumn", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer([]string{"col"}, "mean")
		
		if err := imputer.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Should skip columns without nulls
		stats := imputer.GetStats()
		if len(stats) != 0 {
			t.Error("Should skip columns without nulls")
		}
	})

	t.Run("UnknownStrategy", func(t *testing.T) {
		data := map[string]any{
			"col": []any{1.0, nil, 3.0},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer([]string{"col"}, "unknown_strategy")
		
		err := imputer.Fit(df)
		if err == nil {
			t.Error("Should error on unknown strategy")
		}
	})

	t.Run("NotFittedError", func(t *testing.T) {
		data := map[string]any{
			"col": []any{1.0, nil, 3.0},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer([]string{"col"}, "mean")
		
		_, err := imputer.Transform(df)
		if err == nil {
			t.Error("Transform should fail when not fitted")
		}
	})

	t.Run("AutoDetectColumns", func(t *testing.T) {
		data := map[string]any{
			"col1": []any{1.0, nil, 3.0},
			"col2": []any{10.0, 20.0, nil},
		}
		df, _ := dataframe.New(data)

		imputer := NewSimpleImputer(nil, "mean") // Auto-detect
		
		if err := imputer.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		stats := imputer.GetStats()
		if len(stats) == 0 {
			t.Error("Should auto-detect columns with nulls")
		}
	})
}

func TestSimpleImputerFitTransform(t *testing.T) {
	data := map[string]any{
		"col": []any{1.0, nil, 3.0, nil, 5.0},
	}
	df, _ := dataframe.New(data)

	imputer := NewSimpleImputer([]string{"col"}, "median")
	imputed, err := imputer.FitTransform(df)
	
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if !imputer.IsFitted() {
		t.Error("Imputer should be fitted after FitTransform")
	}

	if imputed.Nrows() != df.Nrows() {
		t.Errorf("Expected %d rows, got %d", df.Nrows(), imputed.Nrows())
	}
}
