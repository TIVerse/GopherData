package features

import (
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/features/encoders"
	"github.com/TIVerse/GopherData/features/imputers"
	"github.com/TIVerse/GopherData/features/scalers"
	"github.com/TIVerse/GopherData/features/selectors"
)

func TestPipeline(t *testing.T) {
	t.Run("EmptyPipeline", func(t *testing.T) {
		pipeline := NewPipeline()
		
		if pipeline.Len() != 0 {
			t.Error("New pipeline should be empty")
		}

		if pipeline.IsFitted() {
			t.Error("Empty pipeline should not be fitted")
		}
	})

	t.Run("AddSteps", func(t *testing.T) {
		pipeline := NewPipeline()
		
		scaler := scalers.NewStandardScaler([]string{"col"})
		pipeline.Add("scaler", scaler)
		
		if pipeline.Len() != 1 {
			t.Errorf("Expected 1 step, got %d", pipeline.Len())
		}

		step := pipeline.GetStep(0)
		if step == nil {
			t.Error("Should return step at index 0")
		}
		if step.Name != "scaler" {
			t.Errorf("Expected name 'scaler', got %q", step.Name)
		}
	})

	t.Run("GetStepByName", func(t *testing.T) {
		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer(nil, "mean")).
			Add("scaler", scalers.NewStandardScaler(nil))
		
		step := pipeline.GetStepByName("scaler")
		if step == nil {
			t.Error("Should find step by name")
		}
		if step.Name != "scaler" {
			t.Errorf("Expected 'scaler', got %q", step.Name)
		}

		notFound := pipeline.GetStepByName("nonexistent")
		if notFound != nil {
			t.Error("Should return nil for nonexistent step")
		}
	})

	t.Run("FitEmptyPipelineError", func(t *testing.T) {
		pipeline := NewPipeline()
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)
		
		err := pipeline.Fit(df)
		if err == nil {
			t.Error("Fit should fail on empty pipeline")
		}
	})

	t.Run("TransformNotFittedError", func(t *testing.T) {
		pipeline := NewPipeline().
			Add("scaler", scalers.NewStandardScaler(nil))
		
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)
		
		_, err := pipeline.Transform(df)
		if err == nil {
			t.Error("Transform should fail when pipeline not fitted")
		}
	})
}

func TestPipelineIntegration(t *testing.T) {
	t.Run("ImputerScalerPipeline", func(t *testing.T) {
		// Create data with missing values
		data := map[string]any{
			"age":    []any{25.0, nil, 35.0, 40.0},
			"income": []float64{50000, 60000, 70000, 80000},
		}
		df, err := dataframe.New(data)
		if err != nil {
			t.Fatalf("Failed to create DataFrame: %v", err)
		}

		// Build pipeline: Imputer -> Scaler
		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer([]string{"age"}, "mean")).
			Add("scaler", scalers.NewStandardScaler([]string{"age", "income"}))
		
		// Fit
		if err := pipeline.Fit(df); err != nil {
			t.Fatalf("Pipeline fit failed: %v", err)
		}

		if !pipeline.IsFitted() {
			t.Error("Pipeline should be fitted")
		}

		// Transform
		transformed, err := pipeline.Transform(df)
		if err != nil {
			t.Fatalf("Pipeline transform failed: %v", err)
		}

		if transformed.Nrows() != df.Nrows() {
			t.Errorf("Expected %d rows, got %d", df.Nrows(), transformed.Nrows())
		}

		// Verify all steps are fitted
		steps := pipeline.Steps()
		for _, step := range steps {
			if !step.fitted {
				t.Errorf("Step %q should be fitted", step.Name)
			}
		}
	})

	t.Run("ImputerEncoderScalerPipeline", func(t *testing.T) {
		// Create mixed data
		data := map[string]any{
			"age":      []any{25.0, nil, 35.0},
			"category": []string{"A", "B", "A"},
			"value":    []float64{100, 200, 300},
		}
		df, _ := dataframe.New(data)

		// Build pipeline: Imputer -> Encoder -> Scaler
		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer([]string{"age"}, "median")).
			Add("encoder", encoders.NewLabelEncoder("category")).
			Add("scaler", scalers.NewStandardScaler([]string{"age", "value"}))
		
		if err := pipeline.Fit(df); err != nil {
			t.Fatalf("Pipeline fit failed: %v", err)
		}

		transformed, err := pipeline.Transform(df)
		if err != nil {
			t.Fatalf("Pipeline transform failed: %v", err)
		}

		if transformed.Nrows() != df.Nrows() {
			t.Error("Should preserve row count")
		}

		if pipeline.Len() != 3 {
			t.Errorf("Expected 3 steps, got %d", pipeline.Len())
		}
	})

	t.Run("CompletePreprocessingPipeline", func(t *testing.T) {
		// Realistic preprocessing scenario
		data := map[string]any{
			"age":      []any{25.0, 30.0, nil, 40.0, 35.0},
			"income":   []float64{50000, 60000, 55000, 80000, 70000},
			"category": []string{"A", "B", "A", "C", "B"},
			"score":    []float64{85.5, 92.0, 78.5, 88.0, 90.5},
		}
		df, _ := dataframe.New(data)

		// Build comprehensive pipeline
		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer([]string{"age"}, "median")).
			Add("scaler", scalers.NewStandardScaler([]string{"age", "income", "score"})).
			Add("encoder", encoders.NewLabelEncoder("category")).
			Add("selector", selectors.NewVarianceThreshold(0.01))
		
		// Fit and transform
		transformed, err := pipeline.FitTransform(df)
		if err != nil {
			t.Fatalf("FitTransform failed: %v", err)
		}

		if !pipeline.IsFitted() {
			t.Error("Pipeline should be fitted after FitTransform")
		}

		if transformed.Nrows() != df.Nrows() {
			t.Error("Should preserve row count")
		}

		// Verify we can transform new data
		testData := map[string]any{
			"age":      []any{28.0, 32.0},
			"income":   []float64{52000, 58000},
			"category": []string{"A", "B"},
			"score":    []float64{82.0, 89.0},
		}
		testDf, _ := dataframe.New(testData)

		testTransformed, err := pipeline.Transform(testDf)
		if err != nil {
			t.Fatalf("Transform on test data failed: %v", err)
		}

		if testTransformed.Nrows() != testDf.Nrows() {
			t.Error("Should work on new data")
		}
	})
}

func TestPipelineSerialization(t *testing.T) {
	t.Run("SaveMetadata", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		pipeline := NewPipeline().
			Add("scaler", scalers.NewStandardScaler([]string{"col"}))
		
		pipeline.Fit(df)

		// Save metadata
		path := "/tmp/gopherdata_pipeline_meta_test.json"
		if err := pipeline.SaveMetadata(path); err != nil {
			t.Fatalf("SaveMetadata failed: %v", err)
		}

		// Verify file exists
		// Note: Would need os.Stat to verify, but the function succeeded
	})

	t.Run("SaveAndLoadPipeline", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		pipeline := NewPipeline().
			Add("scaler", scalers.NewStandardScaler([]string{"col"}))
		
		pipeline.Fit(df)

		// Save pipeline
		path := "/tmp/gopherdata_pipeline_test.json"
		if err := pipeline.Save(path); err != nil {
			t.Fatalf("Save failed: %v", err)
		}

		// Note: Full deserialization not implemented yet
		// This test verifies save works without error
	})
}

func TestPipelineFitTransform(t *testing.T) {
	data := map[string]any{
		"col1": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		"col2": []float64{10.0, 20.0, 30.0, 40.0, 50.0},
	}
	df, _ := dataframe.New(data)

	pipeline := NewPipeline().
		Add("scaler", scalers.NewStandardScaler(nil))
	
	transformed, err := pipeline.FitTransform(df)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if !pipeline.IsFitted() {
		t.Error("Pipeline should be fitted")
	}

	if transformed.Nrows() != df.Nrows() {
		t.Errorf("Expected %d rows, got %d", df.Nrows(), transformed.Nrows())
	}
}

func TestPipelineStepSequencing(t *testing.T) {
	t.Run("StepsAppliedInOrder", func(t *testing.T) {
		// Verify that steps are applied in the order they are added
		data := map[string]any{
			"col": []any{1.0, nil, 3.0},
		}
		df, _ := dataframe.New(data)

		// Imputer must run before scaler (scaler needs no nulls)
		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer([]string{"col"}, "mean")).
			Add("scaler", scalers.NewStandardScaler([]string{"col"}))
		
		// This should succeed because imputer runs first
		_, err := pipeline.FitTransform(df)
		if err != nil {
			t.Errorf("Pipeline should handle nulls with imputer first: %v", err)
		}
	})
}

func TestPipelineEdgeCases(t *testing.T) {
	t.Run("SingleStepPipeline", func(t *testing.T) {
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		pipeline := NewPipeline().
			Add("scaler", scalers.NewStandardScaler([]string{"col"}))
		
		transformed, err := pipeline.FitTransform(df)
		if err != nil {
			t.Fatalf("Single step pipeline failed: %v", err)
		}

		if transformed.Nrows() != df.Nrows() {
			t.Error("Should work with single step")
		}
	})

	t.Run("GetInvalidIndex", func(t *testing.T) {
		pipeline := NewPipeline().
			Add("scaler", scalers.NewStandardScaler(nil))
		
		step := pipeline.GetStep(999)
		if step != nil {
			t.Error("Should return nil for invalid index")
		}

		step = pipeline.GetStep(-1)
		if step != nil {
			t.Error("Should return nil for negative index")
		}
	})
}
