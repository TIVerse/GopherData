// Complete ML Pipeline Example
// This example demonstrates a full machine learning workflow using GopherData
package main

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/features"
	"github.com/TIVerse/GopherData/features/imputers"
	"github.com/TIVerse/GopherData/features/scalers"
	"github.com/TIVerse/GopherData/io/csv"
	"github.com/TIVerse/GopherData/models"
	"github.com/TIVerse/GopherData/models/crossval"
	"github.com/TIVerse/GopherData/models/linear"
	"github.com/TIVerse/GopherData/models/tree"
	"github.com/TIVerse/GopherData/stats"
)

func main() {
	fmt.Println("=== GopherData Complete ML Pipeline ===\n")

	// Step 1: Load Data
	fmt.Println("1. Loading data...")
	data := map[string]any{
		"age":      []float64{25, 30, 35, 40, 45, 50, 55, 60, 65, 70},
		"income":   []float64{30000, 45000, 55000, 65000, 75000, 85000, 95000, 105000, 115000, 125000},
		"category": []string{"A", "B", "A", "B", "A", "B", "A", "B", "A", "B"},
		"score":    []float64{5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0},
	}
	df, _ := dataframe.New(data)
	fmt.Printf("   Loaded %d rows x %d columns\n\n", df.Nrows(), df.Ncols())

	// Step 2: Exploratory Data Analysis
	fmt.Println("2. Exploratory Data Analysis...")
	
	// Descriptive statistics
	ageData := make([]float64, df.Nrows())
	for i := 0; i < df.Nrows(); i++ {
		ageSeries, _ := df.Column("age")
		val, _ := ageSeries.Get(i)
		ageData[i] = val.(float64)
	}
	summary := stats.Describe(ageData)
	fmt.Printf("   Age Statistics:\n")
	fmt.Printf("     Mean: %.2f, Median: %.2f, Std: %.2f\n", 
		summary.Mean, summary.Median, summary.Std)
	
	// Correlation
	incomeData := make([]float64, df.Nrows())
	for i := 0; i < df.Nrows(); i++ {
		incomeSeries, _ := df.Column("income")
		val, _ := incomeSeries.Get(i)
		incomeData[i] = val.(float64)
	}
	scoreData := make([]float64, df.Nrows())
	for i := 0; i < df.Nrows(); i++ {
		scoreSeries, _ := df.Column("score")
		val, _ := scoreSeries.Get(i)
		scoreData[i] = val.(float64)
	}
	
	corr, _ := stats.Pearson(incomeData, scoreData)
	fmt.Printf("   Income-Score Correlation: %.3f\n\n", corr)

	// Step 3: Feature Engineering
	fmt.Println("3. Feature Engineering...")
	
	// Create feature pipeline
	pipeline := features.NewPipeline().
		Add("imputer", imputers.NewSimpleImputer([]string{"age", "income"}, "mean")).
		Add("scaler", scalers.NewStandardScaler([]string{"age", "income"}))
	
	// Fit and transform
	X := df.Select("age", "income")
	y, _ := df.Column("score")
	
	XTransformed, _ := pipeline.FitTransform(X)
	fmt.Printf("   Applied pipeline: imputation + scaling\n")
	fmt.Printf("   Transformed shape: %d x %d\n\n", XTransformed.Nrows(), XTransformed.Ncols())

	// Step 4: Model Training & Evaluation
	fmt.Println("4. Model Training & Evaluation...")
	
	// Train/test split
	split, _ := models.TrainTestSplitFunc(XTransformed, y, 0.2, true, "", 42)
	fmt.Printf("   Train: %d samples, Test: %d samples\n", 
		split.XTrain.Nrows(), split.XTest.Nrows())

	// Train multiple models
	modelsToTry := []struct {
		name  string
		model crossval.Scorer
	}{
		{"Linear Regression", linear.NewLinearRegression(true)},
		{"Ridge Regression", linear.NewRidge(1.0, true)},
		{"Decision Tree", tree.NewDecisionTreeRegressor(5, 2)},
	}

	fmt.Println("\n   Model Comparison:")
	for _, m := range modelsToTry {
		// Train
		m.model.Fit(split.XTrain, split.YTrain)
		
		// Predict
		yPred, _ := m.model.Predict(split.XTest)
		
		// Evaluate
		r2 := models.R2Score(split.YTest, yPred)
		rmse := models.RMSE(split.YTest, yPred)
		
		fmt.Printf("     %s: R²=%.4f, RMSE=%.4f\n", m.name, r2, rmse)
	}

	// Step 5: Cross-Validation
	fmt.Println("\n5. Cross-Validation...")
	
	kf := crossval.NewKFold(3, true, 42)
	model := linear.NewLinearRegression(true)
	
	scores, _ := crossval.CrossValScore(model, XTransformed, y, kf,
		func(yTrue, yPred *core.Series[any]) float64 {
			return models.R2Score(yTrue, yPred)
		})
	
	// Calculate mean score
	meanScore := 0.0
	for _, score := range scores {
		meanScore += score
	}
	meanScore /= float64(len(scores))
	
	fmt.Printf("   3-Fold CV R² Score: %.4f\n", meanScore)

	// Step 6: Final Model
	fmt.Println("\n6. Training Final Model...")
	
	finalModel := linear.NewLinearRegression(true)
	finalModel.Fit(XTransformed, y)
	
	fmt.Printf("   Model Coefficients: %v\n", finalModel.Coef())
	fmt.Printf("   Intercept: %.4f\n", finalModel.Intercept())

	// Step 7: Save Results
	fmt.Println("\n7. Saving Results...")
	
	// Save pipeline
	err := pipeline.Save("model_pipeline.json")
	if err != nil {
		fmt.Printf("   Pipeline saved to model_pipeline.json\n")
	}
	
	// Save transformed data
	err = csv.ToCSV(XTransformed, "transformed_features.csv")
	if err != nil {
		fmt.Printf("   Transformed features saved to transformed_features.csv\n")
	}

	fmt.Println("\n=== Pipeline Complete! ===")
}
