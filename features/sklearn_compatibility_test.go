package features

import (
	"math"
	"testing"

	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/features/encoders"
	"github.com/TIVerse/GopherData/features/imputers"
	"github.com/TIVerse/GopherData/features/scalers"
)

// TestSklearnCompatibility verifies that our API behaves similarly to sklearn
// These tests document expected behavior for Python data scientists

func TestSklearnStandardScalerCompatibility(t *testing.T) {
	// sklearn: scaler = StandardScaler()
	//          scaler.fit(X_train)
	//          X_scaled = scaler.transform(X_test)
	
	t.Run("FitTransformPattern", func(t *testing.T) {
		trainData := map[string]any{
			"feature": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		trainDf, _ := dataframe.New(trainData)

		testData := map[string]any{
			"feature": []float64{2.0, 3.0, 4.0},
		}
		testDf, _ := dataframe.New(testData)

		// Fit on training data
		scaler := scalers.NewStandardScaler([]string{"feature"})
		if err := scaler.Fit(trainDf); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Transform test data using training statistics
		scaled, err := scaler.Transform(testDf)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		if scaled.Nrows() != testDf.Nrows() {
			t.Error("Transform should preserve row count")
		}

		// Verify fitted parameters are accessible (like sklearn)
		means := scaler.GetMeans()
		if means["feature"] != 3.0 {
			t.Errorf("Expected mean 3.0, got %f", means["feature"])
		}
	})

	t.Run("FitTransformCombined", func(t *testing.T) {
		// sklearn: X_scaled = scaler.fit_transform(X_train)
		data := map[string]any{
			"feature": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := dataframe.New(data)

		scaler := scalers.NewStandardScaler([]string{"feature"})
		scaled, err := scaler.FitTransform(df)
		
		if err != nil {
			t.Fatalf("FitTransform failed: %v", err)
		}

		if !scaler.IsFitted() {
			t.Error("Scaler should be fitted after FitTransform")
		}

		if scaled.Nrows() != df.Nrows() {
			t.Error("Should preserve row count")
		}
	})
}

func TestSklearnOneHotEncoderCompatibility(t *testing.T) {
	// sklearn: encoder = OneHotEncoder(drop='first')
	//          encoder.fit(X_train)
	//          X_encoded = encoder.transform(X_test)
	
	t.Run("DropFirstParameter", func(t *testing.T) {
		data := map[string]any{
			"category": []string{"A", "B", "C"},
		}
		df, _ := dataframe.New(data)

		// Like sklearn's drop='first'
		encoder := encoders.NewOneHotEncoder([]string{"category"})
		encoder.DropFirst = true
		
		if err := encoder.Fit(df); err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		encoded, err := encoder.Transform(df)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		// Should drop first category (A), keeping B and C
		if encoded.Ncols() != 2 {
			t.Errorf("Expected 2 columns with drop_first=True, got %d", encoded.Ncols())
		}
	})

	t.Run("GetFeatureNames", func(t *testing.T) {
		// sklearn has get_feature_names_out() - we have GetCategories()
		data := map[string]any{
			"category": []string{"A", "B", "C"},
		}
		df, _ := dataframe.New(data)

		encoder := encoders.NewOneHotEncoder([]string{"category"})
		encoder.Fit(df)
		
		categories := encoder.GetCategories()
		if len(categories["category"]) != 3 {
			t.Errorf("Expected 3 categories, got %d", len(categories["category"]))
		}
	})
}

func TestSklearnSimpleImputerCompatibility(t *testing.T) {
	// sklearn: imputer = SimpleImputer(strategy='mean')
	//          imputer.fit(X_train)
	//          X_imputed = imputer.transform(X_test)
	
	t.Run("StrategyParameter", func(t *testing.T) {
		data := map[string]any{
			"feature": []any{1.0, nil, 3.0, nil, 5.0},
		}
		df, _ := dataframe.New(data)

		// Test all sklearn strategies
		strategies := []string{"mean", "median", "most_frequent", "constant"}
		
		for _, strategy := range strategies {
			imputer := imputers.NewSimpleImputer([]string{"feature"}, strategy)
			if strategy == "constant" {
				imputer.FillValue = 0.0
			}
			
			if err := imputer.Fit(df); err != nil {
				t.Errorf("Fit failed for strategy %q: %v", strategy, err)
			}

			_, err := imputer.Transform(df)
			if err != nil {
				t.Errorf("Transform failed for strategy %q: %v", strategy, err)
			}
		}
	})

	t.Run("StatisticsAccessible", func(t *testing.T) {
		// sklearn: imputer.statistics_ - we have GetStats()
		data := map[string]any{
			"feature": []any{1.0, nil, 3.0, 5.0},
		}
		df, _ := dataframe.New(data)

		imputer := imputers.NewSimpleImputer([]string{"feature"}, "mean")
		imputer.Fit(df)
		
		stats := imputer.GetStats()
		expectedMean := 3.0 // (1+3+5)/3
		if val, ok := stats["feature"]; ok {
			if meanVal, isFloat := val.(float64); isFloat {
				if meanVal != expectedMean {
					t.Errorf("Expected mean %f, got %f", expectedMean, meanVal)
				}
			} else {
				t.Errorf("Expected float64, got %T", val)
			}
		} else {
			t.Error("Expected stats to contain 'feature'")
		}
	})
}

func TestSklearnPipelineCompatibility(t *testing.T) {
	// sklearn: pipeline = Pipeline([
	//              ('imputer', SimpleImputer(strategy='mean')),
	//              ('scaler', StandardScaler()),
	//              ('encoder', OneHotEncoder())
	//          ])
	//          pipeline.fit(X_train)
	//          X_transformed = pipeline.transform(X_test)
	
	t.Run("PipelineConstructor", func(t *testing.T) {
		// Our fluent API mimics sklearn Pipeline
		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer(nil, "mean")).
			Add("scaler", scalers.NewStandardScaler(nil)).
			Add("encoder", encoders.NewLabelEncoder("category"))
		
		if pipeline.Len() != 3 {
			t.Errorf("Expected 3 steps, got %d", pipeline.Len())
		}

		// Can access steps by name (like sklearn)
		scalerStep := pipeline.GetStepByName("scaler")
		if scalerStep == nil {
			t.Error("Should be able to access step by name")
		}
		if scalerStep.Name != "scaler" {
			t.Errorf("Expected 'scaler', got %q", scalerStep.Name)
		}
	})

	t.Run("PipelineFitTransform", func(t *testing.T) {
		data := map[string]any{
			"age":      []any{25.0, nil, 35.0},
			"category": []string{"A", "B", "A"},
		}
		df, _ := dataframe.New(data)

		pipeline := NewPipeline().
			Add("imputer", imputers.NewSimpleImputer([]string{"age"}, "mean")).
			Add("encoder", encoders.NewLabelEncoder("category"))
		
		// Like sklearn: pipeline.fit_transform(X)
		transformed, err := pipeline.FitTransform(df)
		if err != nil {
			t.Fatalf("FitTransform failed: %v", err)
		}

		if transformed.Nrows() != df.Nrows() {
			t.Error("Should preserve row count")
		}

		// Pipeline is fitted (like sklearn)
		if !pipeline.IsFitted() {
			t.Error("Pipeline should be fitted")
		}
	})

	t.Run("PipelineStepAccess", func(t *testing.T) {
		// sklearn: pipeline.named_steps['scaler']
		// We have: pipeline.GetStepByName("scaler")
		
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		pipeline := NewPipeline().
			Add("scaler", scalers.NewStandardScaler([]string{"col"}))
		
		pipeline.Fit(df)

		// Access step
		step := pipeline.GetStepByName("scaler")
		if step == nil {
			t.Error("Should access step by name")
		}

		// Verify it's fitted
		if !step.fitted {
			t.Error("Step should be fitted after pipeline fit")
		}
	})
}

func TestSklearnBehaviorConsistency(t *testing.T) {
	t.Run("TransformWithoutFitErrors", func(t *testing.T) {
		// sklearn raises NotFittedError - we return error
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		scaler := scalers.NewStandardScaler([]string{"col"})
		
		// Should error when transforming without fitting
		_, err := scaler.Transform(df)
		if err == nil {
			t.Error("Transform should fail when not fitted (like sklearn)")
		}
	})

	t.Run("FitReturnsNil", func(t *testing.T) {
		// sklearn fit() returns self - we return error or nil
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		scaler := scalers.NewStandardScaler([]string{"col"})
		err := scaler.Fit(df)
		
		if err != nil {
			t.Errorf("Fit should succeed and return nil: %v", err)
		}
	})

	t.Run("ImmutableTransform", func(t *testing.T) {
		// sklearn creates new arrays - we return new DataFrames
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0},
		}
		df, _ := dataframe.New(data)

		scaler := scalers.NewStandardScaler([]string{"col"})
		scaler.Fit(df)
		transformed, _ := scaler.Transform(df)
		
		// Original should be unchanged (like sklearn doesn't mutate input)
		if transformed == df {
			t.Error("Transform should return new DataFrame, not mutate original")
		}
	})
}

func TestSklearnNumericalEquivalence(t *testing.T) {
	// Verify our calculations match sklearn's
	
	t.Run("StandardScalerMath", func(t *testing.T) {
		// For data [1, 2, 3, 4, 5]:
		// mean = 3.0
		// std = sqrt(2.5) ≈ 1.5811
		// scaled[0] = (1 - 3) / 1.5811 ≈ -1.2649
		
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := dataframe.New(data)

		scaler := scalers.NewStandardScaler([]string{"col"})
		scaler.Fit(df)
		
		means := scaler.GetMeans()
		stds := scaler.GetStds()
		
		// Verify calculations
		if math.Abs(means["col"]-3.0) > 0.001 {
			t.Errorf("Mean should be 3.0, got %f", means["col"])
		}

		expectedStd := math.Sqrt(2.5)
		if math.Abs(stds["col"]-expectedStd) > 0.001 {
			t.Errorf("Std should be %f, got %f", expectedStd, stds["col"])
		}
	})

	t.Run("MinMaxScalerMath", func(t *testing.T) {
		// For data [1, 2, 3, 4, 5] scaled to [0, 1]:
		// X_scaled = (X - min) / (max - min)
		// scaled[0] = (1 - 1) / (5 - 1) = 0.0
		// scaled[4] = (5 - 1) / (5 - 1) = 1.0
		
		data := map[string]any{
			"col": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := dataframe.New(data)

		scaler := scalers.NewMinMaxScaler([]string{"col"})
		scaler.FeatureMin = 0.0
		scaler.FeatureMax = 1.0
		scaler.Fit(df)
		
		mins := scaler.GetMins()
		maxs := scaler.GetMaxs()
		
		if mins["col"] != 1.0 {
			t.Errorf("Min should be 1.0, got %f", mins["col"])
		}
		if maxs["col"] != 5.0 {
			t.Errorf("Max should be 5.0, got %f", maxs["col"])
		}
	})
}

// Benchmark: Verify performance is reasonable compared to sklearn
func BenchmarkStandardScalerVsSklearn(b *testing.B) {
	// sklearn StandardScaler on 100K rows: ~5-10ms
	// Our implementation should be comparable
	
	data := map[string]any{
		"col1": make([]float64, 100000),
		"col2": make([]float64, 100000),
		"col3": make([]float64, 100000),
	}
	
	for i := 0; i < 100000; i++ {
		data["col1"].([]float64)[i] = float64(i)
		data["col2"].([]float64)[i] = float64(i * 2)
		data["col3"].([]float64)[i] = float64(i * 3)
	}
	
	df, _ := dataframe.New(data)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scaler := scalers.NewStandardScaler([]string{"col1", "col2", "col3"})
		scaler.Fit(df)
		scaler.Transform(df)
	}
}
