package main

import (
	"fmt"
	"log"

	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/features"
	"github.com/TIVerse/GopherData/features/creators"
	"github.com/TIVerse/GopherData/features/encoders"
	"github.com/TIVerse/GopherData/features/imputers"
	"github.com/TIVerse/GopherData/features/scalers"
	"github.com/TIVerse/GopherData/features/selectors"
)

func main() {
	fmt.Println("=== GopherData Phase 3: Feature Engineering Demo ===\n")

	// Create sample dataset
	data := map[string]any{
		"age":      []any{25, 30, nil, 40, 35, 28, 32, 45, nil, 38},
		"income":   []float64{50000, 60000, 55000, 80000, 70000, 52000, 58000, 90000, 65000, 75000},
		"category": []string{"A", "B", "A", "C", "B", "A", "B", "C", "A", "B"},
		"score":    []float64{85.5, 92.0, 78.5, 88.0, 90.5, 82.0, 89.0, 95.0, 80.0, 87.5},
		"target":   []float64{1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0},
	}

	df, err := dataframe.New(data)
	if err != nil {
		log.Fatalf("Failed to create DataFrame: %v", err)
	}

	fmt.Println("Original Data:")
	fmt.Println(df.Head(5))
	fmt.Println()

	// =============================================================================
	// 1. Missing Data Imputation
	// =============================================================================
	fmt.Println("=== 1. Missing Data Imputation ===")
	
	imputer := imputers.NewSimpleImputer([]string{"age"}, "median")
	if err := imputer.Fit(df); err != nil {
		log.Fatalf("Imputer fit failed: %v", err)
	}
	
	df, err = imputer.Transform(df)
	if err != nil {
		log.Fatalf("Imputer transform failed: %v", err)
	}
	
	fmt.Println("After imputation:")
	fmt.Println(df.Head(5))
	fmt.Println()

	// =============================================================================
	// 2. Feature Scaling
	// =============================================================================
	fmt.Println("=== 2. Feature Scaling ===")
	
	scaler := scalers.NewStandardScaler([]string{"age", "income", "score"})
	if err := scaler.Fit(df); err != nil {
		log.Fatalf("Scaler fit failed: %v", err)
	}
	
	scaledDf, err := scaler.Transform(df)
	if err != nil {
		log.Fatalf("Scaler transform failed: %v", err)
	}
	
	fmt.Println("After standard scaling:")
	fmt.Println(scaledDf.Head(3))
	fmt.Printf("Means: %v\n", scaler.GetMeans())
	fmt.Printf("Stds: %v\n", scaler.GetStds())
	fmt.Println()

	// =============================================================================
	// 3. Categorical Encoding
	// =============================================================================
	fmt.Println("=== 3. Categorical Encoding ===")
	
	// Label Encoding
	labelEncoder := encoders.NewLabelEncoder("category")
	if err := labelEncoder.Fit(df); err != nil {
		log.Fatalf("Label encoder fit failed: %v", err)
	}
	
	labelEncodedDf, err := labelEncoder.Transform(df)
	if err != nil {
		log.Fatalf("Label encoder transform failed: %v", err)
	}
	
	fmt.Println("After label encoding:")
	fmt.Println(labelEncodedDf.Head(3))
	fmt.Printf("Mapping: %v\n", labelEncoder.GetMapping())
	fmt.Println()

	// One-Hot Encoding
	onehotEncoder := encoders.NewOneHotEncoder([]string{"category"})
	onehotEncoder.DropFirst = true // Avoid multicollinearity
	
	if err := onehotEncoder.Fit(df); err != nil {
		log.Fatalf("OneHot encoder fit failed: %v", err)
	}
	
	onehotDf, err := onehotEncoder.Transform(df)
	if err != nil {
		log.Fatalf("OneHot encoder transform failed: %v", err)
	}
	
	fmt.Println("After one-hot encoding (drop_first=True):")
	fmt.Println(onehotDf.Head(3))
	fmt.Println()

	// =============================================================================
	// 4. Feature Selection
	// =============================================================================
	fmt.Println("=== 4. Feature Selection ===")
	
	// Variance threshold
	varSelector := selectors.NewVarianceThreshold(0.01)
	if err := varSelector.Fit(df); err != nil {
		log.Fatalf("Variance selector fit failed: %v", err)
	}
	
	selectedDf, err := varSelector.Transform(df)
	if err != nil {
		log.Fatalf("Variance selector transform failed: %v", err)
	}
	
	fmt.Printf("Features selected by variance threshold: %v\n", varSelector.GetSelectedFeatures())
	fmt.Printf("Shape after selection: (%d, %d)\n", selectedDf.Nrows(), selectedDf.Ncols())
	fmt.Println()

	// K-Best selection
	kbest := selectors.NewSelectKBest(3)
	if err := kbest.Fit(df, "target"); err != nil {
		log.Fatalf("K-Best selector fit failed: %v", err)
	}
	
	kbestDf, err := kbest.Transform(df)
	if err != nil {
		log.Fatalf("K-Best selector transform failed: %v", err)
	}
	
	fmt.Printf("Top 3 features by correlation: %v\n", kbest.GetSelectedFeatures())
	fmt.Printf("Feature scores: %v\n", kbest.GetScores())
	fmt.Println()

	// =============================================================================
	// 5. Feature Creation
	// =============================================================================
	fmt.Println("=== 5. Feature Creation ===")
	
	// Polynomial features
	poly := creators.NewPolynomialFeatures(2)
	poly.Columns = []string{"age", "income"}
	poly.InteractionOnly = false
	
	if err := poly.Fit(df); err != nil {
		log.Fatalf("Polynomial features fit failed: %v", err)
	}
	
	polyDf, err := poly.Transform(df)
	if err != nil {
		log.Fatalf("Polynomial features transform failed: %v", err)
	}
	
	fmt.Printf("After polynomial features (degree=2): %d columns\n", polyDf.Ncols())
	fmt.Printf("New columns include: age^2, age*income, income^2\n")
	fmt.Println()

	// Interaction features
	interaction := creators.NewInteractionFeatures([]string{"age", "income", "score"})
	if err := interaction.Fit(df); err != nil {
		log.Fatalf("Interaction features fit failed: %v", err)
	}
	
	interactionDf, err := interaction.Transform(df)
	if err != nil {
		log.Fatalf("Interaction features transform failed: %v", err)
	}
	
	fmt.Printf("After pairwise interactions: %d columns\n", interactionDf.Ncols())
	fmt.Printf("New columns include: age*income, age*score, income*score\n")
	fmt.Println()

	// =============================================================================
	// 6. Complete Pipeline
	// =============================================================================
	fmt.Println("=== 6. Complete Feature Engineering Pipeline ===")
	
	// Build a comprehensive pipeline
	pipeline := features.NewPipeline().
		Add("imputer", imputers.NewSimpleImputer([]string{"age"}, "median")).
		Add("scaler", scalers.NewStandardScaler([]string{"age", "income", "score"})).
		Add("encoder", encoders.NewLabelEncoder("category")).
		Add("selector", selectors.NewVarianceThreshold(0.01))
	
	fmt.Printf("Pipeline has %d steps\n", pipeline.Len())
	
	// Fit the entire pipeline
	if err := pipeline.Fit(df); err != nil {
		log.Fatalf("Pipeline fit failed: %v", err)
	}
	
	fmt.Println("Pipeline fitted successfully!")
	
	// Transform the data
	transformedDf, err := pipeline.Transform(df)
	if err != nil {
		log.Fatalf("Pipeline transform failed: %v", err)
	}
	
	fmt.Printf("Final shape: (%d, %d)\n", transformedDf.Nrows(), transformedDf.Ncols())
	fmt.Println("Transformed data:")
	fmt.Println(transformedDf.Head(5))
	fmt.Println()

	// =============================================================================
	// 7. Pipeline Serialization
	// =============================================================================
	fmt.Println("=== 7. Pipeline Serialization ===")
	
	// Save pipeline
	pipelinePath := "/tmp/gopherdata_pipeline.json"
	if err := pipeline.Save(pipelinePath); err != nil {
		log.Fatalf("Pipeline save failed: %v", err)
	}
	
	fmt.Printf("Pipeline saved to %s\n", pipelinePath)
	
	// Save metadata
	metadataPath := "/tmp/gopherdata_pipeline_metadata.json"
	if err := pipeline.SaveMetadata(metadataPath); err != nil {
		log.Fatalf("Metadata save failed: %v", err)
	}
	
	fmt.Printf("Metadata saved to %s\n", metadataPath)
	fmt.Println()

	// =============================================================================
	// 8. Different Scalers Comparison
	// =============================================================================
	fmt.Println("=== 8. Scalers Comparison ===")
	
	// MinMax Scaler
	minmaxScaler := scalers.NewMinMaxScaler([]string{"age", "income"})
	minmaxScaler.FeatureMin = 0.0
	minmaxScaler.FeatureMax = 1.0
	
	if err := minmaxScaler.Fit(df); err == nil {
		minmaxDf, _ := minmaxScaler.Transform(df)
		fmt.Printf("MinMax scaler: range [%.1f, %.1f]\n", minmaxScaler.FeatureMin, minmaxScaler.FeatureMax)
		fmt.Printf("  Shape: (%d, %d)\n", minmaxDf.Nrows(), minmaxDf.Ncols())
	}
	
	// Robust Scaler
	robustScaler := scalers.NewRobustScaler([]string{"age", "income"})
	if err := robustScaler.Fit(df); err == nil {
		robustDf, _ := robustScaler.Transform(df)
		fmt.Printf("Robust scaler: uses median and IQR\n")
		fmt.Printf("  Shape: (%d, %d)\n", robustDf.Nrows(), robustDf.Ncols())
	}
	
	// MaxAbs Scaler
	maxabsScaler := scalers.NewMaxAbsScaler([]string{"age", "income"})
	if err := maxabsScaler.Fit(df); err == nil {
		maxabsDf, _ := maxabsScaler.Transform(df)
		fmt.Printf("MaxAbs scaler: preserves sparsity\n")
		fmt.Printf("  Shape: (%d, %d)\n", maxabsDf.Nrows(), maxabsDf.Ncols())
	}
	
	fmt.Println()

	// =============================================================================
	// Summary
	// =============================================================================
	fmt.Println("=== Summary ===")
	fmt.Println("✓ Imputation: Handled missing values")
	fmt.Println("✓ Scaling: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler")
	fmt.Println("✓ Encoding: LabelEncoder, OneHotEncoder")
	fmt.Println("✓ Selection: VarianceThreshold, SelectKBest")
	fmt.Println("✓ Creation: PolynomialFeatures, InteractionFeatures")
	fmt.Println("✓ Pipeline: Chained transformations")
	fmt.Println("✓ Serialization: Save/load pipelines")
	fmt.Println()
	fmt.Println("Phase 3 Feature Engineering Complete! 🎉")
}
