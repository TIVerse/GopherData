package encoders

import (
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
)

func TestOneHotEncoder(t *testing.T) {
	t.Run("BasicEncoding", func(t *testing.T) {
		data := map[string]any{
			"category": []string{"A", "B", "C", "A", "B"},
			"value":    []int64{1, 2, 3, 4, 5},
		}
		df, err := dataframe.New(data)
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		encoder := NewOneHotEncoder([]string{"category"})
		
		// Fit
		if err := encoder.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		if !encoder.IsFitted() {
			t.Error("Encoder should be fitted")
		}

		// Check learned categories
		categories := encoder.GetCategories()
		if len(categories["category"]) != 3 {
			t.Errorf("Expected 3 categories, got %d", len(categories["category"]))
		}

		// Transform
		encoded, err := encoder.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		// Should have 3 new columns (category_A, category_B, category_C)
		// and drop original, plus keep 'value' column
		expectedCols := 4 // value + 3 binary columns
		if encoded.Ncols() != expectedCols {
			t.Errorf("Expected %d columns, got %d", expectedCols, encoded.Ncols())
		}

		if encoded.Nrows() != df.Nrows() {
			t.Errorf("Expected %d rows, got %d", df.Nrows(), encoded.Nrows())
		}
	})

	t.Run("DropFirstTrue", func(t *testing.T) {
		data := map[string]any{
			"category": []string{"A", "B", "C"},
		}
		df, _ := dataframe.New(data)

		encoder := NewOneHotEncoder([]string{"category"})
		encoder.DropFirst = true
		
		if err := encoder.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		encoded, err := encoder.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		// Should have 2 binary columns (dropping first)
		if encoded.Ncols() != 2 {
			t.Errorf("Expected 2 columns with DropFirst=true, got %d", encoded.Ncols())
		}
	})

	t.Run("WithNulls", func(t *testing.T) {
		data := map[string]any{
			"category": []any{"A", nil, "B", "A"},
		}
		df, _ := dataframe.New(data)

		encoder := NewOneHotEncoder([]string{"category"})
		
		if err := encoder.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		encoded, err := encoder.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if encoded.Nrows() != df.Nrows() {
			t.Error("Should preserve all rows including nulls")
		}
	})

	t.Run("MultipleColumns", func(t *testing.T) {
		data := map[string]any{
			"cat1": []string{"A", "B"},
			"cat2": []string{"X", "Y"},
		}
		df, _ := dataframe.New(data)

		encoder := NewOneHotEncoder([]string{"cat1", "cat2"})
		
		if err := encoder.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		categories := encoder.GetCategories()
		if len(categories) != 2 {
			t.Errorf("Expected 2 column mappings, got %d", len(categories))
		}

		encoded, err := encoder.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		// 2 categories per column = 4 binary columns
		if encoded.Ncols() != 4 {
			t.Errorf("Expected 4 columns, got %d", encoded.Ncols())
		}
	})

	t.Run("NotFittedError", func(t *testing.T) {
		data := map[string]any{
			"category": []string{"A", "B"},
		}
		df, _ := dataframe.New(data)

		encoder := NewOneHotEncoder([]string{"category"})
		
		_, err := encoder.Transform(df)
		if err == nil {
			t.Error("Transform should fail when not fitted")
		}
	})

	t.Run("NoColumnsError", func(t *testing.T) {
		data := map[string]any{
			"value": []int64{1, 2, 3},
		}
		df, _ := dataframe.New(data)

		encoder := NewOneHotEncoder([]string{})
		
		err := encoder.Fit(df)
		if err == nil {
			t.Error("Fit should fail when no columns specified")
		}
	})
}

func TestOneHotEncoderFitTransform(t *testing.T) {
	data := map[string]any{
		"category": []string{"A", "B", "C"},
	}
	df, _ := dataframe.New(data)

	encoder := NewOneHotEncoder([]string{"category"})
	encoded, err := encoder.FitTransform(df)
	
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if !encoder.IsFitted() {
		t.Error("Encoder should be fitted after FitTransform")
	}

	if encoded.Ncols() != 3 {
		t.Errorf("Expected 3 columns, got %d", encoded.Ncols())
	}
}
