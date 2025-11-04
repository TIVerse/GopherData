// Package gopherdata provides a comprehensive data science library for Go.
//
// GopherData brings pandas-like DataFrame operations and scikit-learn compatible
// machine learning to the Go ecosystem, with a focus on type safety, performance,
// and memory efficiency.
//
// # Core Features
//
// - DataFrames & Series: Pandas-like data structures with type safety
// - Feature Engineering: 22 transformers (scalers, encoders, imputers, selectors, creators)
// - Machine Learning: 16 algorithms including linear models, trees, clustering, and dimensionality reduction
// - Statistics: Descriptive stats, correlation analysis, hypothesis testing, probability distributions
// - I/O Operations: CSV and JSON support with automatic type inference
// - High Performance: Optimized operations with 2-5x speedup over pandas in many cases
//
// # Quick Start
//
// Creating and manipulating DataFrames:
//
//	import (
//	    "github.com/TIVerse/GopherData/dataframe"
//	    "github.com/TIVerse/GopherData/io/csv"
//	)
//
//	// Create a DataFrame
//	df, _ := dataframe.New(map[string]any{
//	    "name":   []string{"Alice", "Bob", "Charlie"},
//	    "age":    []int64{25, 30, 35},
//	    "salary": []float64{50000, 65000, 75000},
//	})
//
//	// Select columns
//	subset := df.Select("name", "age")
//
//	// Filter rows
//	filtered := df.Filter(func(row *dataframe.Row) bool {
//	    age, _ := row.Get("age")
//	    return age.(int64) > 25
//	})
//
//	// GroupBy and aggregate
//	grouped := df.GroupBy("department").Agg(map[string]string{
//	    "salary": "mean",
//	    "age":    "max",
//	})
//
//	// Write to CSV
//	csv.ToCSV(grouped, "output.csv")
//
// # Machine Learning
//
// Training a regression model:
//
//	import (
//	    "github.com/TIVerse/GopherData/models"
//	    "github.com/TIVerse/GopherData/models/linear"
//	)
//
//	// Load data
//	df, _ := csv.ReadCSV("data.csv")
//	X := df.Select("feature1", "feature2", "feature3")
//	y, _ := df.Column("target")
//
//	// Train/test split
//	split, _ := models.TrainTestSplitFunc(X, y, 0.2, true, "", 42)
//
//	// Train model
//	model := linear.NewRidge(1.0, true)
//	model.Fit(split.XTrain, split.YTrain)
//
//	// Predict
//	yPred, _ := model.Predict(split.XTest)
//
//	// Evaluate
//	r2 := models.R2Score(split.YTest, yPred)
//
// # Feature Engineering
//
// Building a preprocessing pipeline:
//
//	import (
//	    "github.com/TIVerse/GopherData/features"
//	    "github.com/TIVerse/GopherData/features/scalers"
//	    "github.com/TIVerse/GopherData/features/encoders"
//	    "github.com/TIVerse/GopherData/features/imputers"
//	)
//
//	// Create preprocessing pipeline
//	pipeline := features.NewPipeline().
//	    Add("imputer", imputers.NewSimpleImputer([]string{"age", "income"}, "mean")).
//	    Add("scaler", scalers.NewStandardScaler([]string{"age", "income"})).
//	    Add("encoder", encoders.NewOneHotEncoder([]string{"category"}))
//
//	// Fit and transform
//	XTransformed, _ := pipeline.FitTransform(df)
//
//	// Save pipeline for later use
//	pipeline.Save("pipeline.json")
//
// # Statistics
//
// Statistical analysis and hypothesis testing:
//
//	import (
//	    "github.com/TIVerse/GopherData/stats"
//	    "github.com/TIVerse/GopherData/stats/hypothesis"
//	)
//
//	data := []float64{23, 25, 28, 29, 30, 32, 35, 37, 38, 40}
//
//	// Descriptive statistics
//	summary := stats.Describe(data)
//
//	// Hypothesis testing
//	tResult := hypothesis.TTest(data, 30)
//
//	// Correlation analysis
//	x := []float64{1, 2, 3, 4, 5}
//	y := []float64{2, 4, 5, 4, 5}
//	corr := stats.Pearson(x, y)
//
// # Package Organization
//
// The library is organized into the following packages:
//
// - core: Core types and interfaces
// - series: Series (1D labeled arrays) implementation
// - dataframe: DataFrame (2D labeled tables) implementation
// - io/csv: CSV reading and writing
// - io/json: JSON reading and writing
// - features: Feature engineering transformers and pipelines
// - features/scalers: Data scaling transformers
// - features/encoders: Categorical encoding transformers
// - features/imputers: Missing value imputation
// - features/selectors: Feature selection methods
// - features/creators: Feature creation methods
// - models: Machine learning models and utilities
// - models/linear: Linear models (LinearRegression, Ridge, Lasso, LogisticRegression)
// - models/tree: Tree-based models (DecisionTree)
// - models/cluster: Clustering algorithms (KMeans)
// - models/decomposition: Dimensionality reduction (PCA)
// - models/crossval: Cross-validation utilities
// - stats: Statistical functions
// - stats/hypothesis: Hypothesis testing
// - stats/distributions: Probability distributions
//
// # Design Principles
//
// - Type Safety: Leverages Go generics for compile-time type checking
// - Memory Efficiency: BitSet-based null handling, copy-on-write semantics
// - Performance: Parallel operations, efficient algorithms, zero-copy views
// - sklearn Compatibility: Familiar API for data scientists migrating from Python
// - Pure Go: No C dependencies for easy deployment
//
// For more information, visit: https://github.com/TIVerse/GopherData
package gopherdata
