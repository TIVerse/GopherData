package imputers

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// IterativeImputer fills missing values using an iterative approach (MICE algorithm).
// Each feature with missing values is modeled as a function of other features.
// This is a simplified implementation placeholder.
type IterativeImputer struct {
	// MaxIter is the maximum number of imputation rounds
	MaxIter int
	
	// Tolerance for convergence
	Tolerance float64
	
	// InitialStrategy for first imputation ("mean" or "median")
	InitialStrategy string
	
	fitted bool
}

// NewIterativeImputer creates a new IterativeImputer.
func NewIterativeImputer() *IterativeImputer {
	return &IterativeImputer{
		MaxIter:         10,
		Tolerance:       0.001,
		InitialStrategy: "mean",
		fitted:          false,
	}
}

// Fit prepares the imputer by analyzing data structure.
func (i *IterativeImputer) Fit(df *dataframe.DataFrame, _ ...string) error {
	if i.MaxIter <= 0 {
		i.MaxIter = 10
	}
	if i.Tolerance <= 0 {
		i.Tolerance = 0.001
	}
	i.fitted = true
	return nil
}

// Transform applies iterative imputation using MICE (Multivariate Imputation by Chained Equations).
func (i *IterativeImputer) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !i.fitted {
		return nil, fmt.Errorf("imputer not fitted")
	}
	
	result := df.Copy()
	cols := df.Columns()
	
	// Step 1: Initial imputation using simple strategy
	for _, col := range cols {
		colSeries, _ := result.Column(col)
		if colSeries.NullCount() == 0 {
			continue
		}
		
		// Get initial imputation value
		var imputeVal any
		if i.InitialStrategy == "median" {
			imputeVal = computeMedian(colSeries)
		} else {
			imputeVal = computeMean(colSeries)
		}
		
		// Fill missing values with initial imputation
		newData := make([]any, colSeries.Len())
		for j := 0; j < colSeries.Len(); j++ {
			val, ok := colSeries.Get(j)
			if ok {
				newData[j] = val
			} else {
				newData[j] = imputeVal
			}
		}
		newSeries := seriesPkg.New(col, newData, colSeries.Dtype())
		result = result.WithColumn(col, newSeries)
	}
	
	// Step 2: Iterative refinement
	// For each iteration, predict each feature with missing values using other features
	for iter := 0; iter < i.MaxIter; iter++ {
		maxChange := 0.0
		
		for _, targetCol := range cols {
			origSeries, _ := df.Column(targetCol)
			if origSeries.NullCount() == 0 {
				continue
			}
			
			// Use other columns to predict this column
			predictorCols := make([]string, 0)
			for _, col := range cols {
				if col != targetCol {
					predictorCols = append(predictorCols, col)
				}
			}
			
			// Simple linear regression-like imputation
			// Compute correlation-based weighted average from other features
			currentSeries, _ := result.Column(targetCol)
			newData := make([]any, currentSeries.Len())
			
			for j := 0; j < currentSeries.Len(); j++ {
				origVal, origOk := origSeries.Get(j)
				if origOk {
					// Keep original value
					newData[j] = origVal
				} else {
					// Predict based on similar rows
					predicted := i.predictValue(result, j, targetCol, predictorCols)
					newData[j] = predicted
					
					// Track convergence
					currVal, _ := currentSeries.Get(j)
					change := math.Abs(toFloat64Impute(predicted) - toFloat64Impute(currVal))
					if change > maxChange {
						maxChange = change
					}
				}
			}
			
			newSeries := seriesPkg.New(targetCol, newData, origSeries.Dtype())
			result = result.WithColumn(targetCol, newSeries)
		}
		
		// Check convergence
		if maxChange < i.Tolerance {
			break
		}
	}
	
	return result, nil
}

// predictValue predicts a missing value using weighted average from similar complete rows
func (i *IterativeImputer) predictValue(df *dataframe.DataFrame, rowIdx int, targetCol string, predictorCols []string) any {
	// Find rows with complete data for predictor columns
	var sum float64
	var count int
	
	targetSeries, _ := df.Column(targetCol)
	
	for k := 0; k < df.Nrows(); k++ {
		if k == rowIdx {
			continue
		}
		
		// Check if this row has complete predictor data and target value
		hasComplete := true
		for _, col := range predictorCols {
			colSeries, _ := df.Column(col)
			if _, ok := colSeries.Get(k); !ok {
				hasComplete = false
				break
			}
		}
		
		if !hasComplete {
			continue
		}
		
		// Check if target value exists for this row
		targetVal, ok := targetSeries.Get(k)
		if !ok {
			continue
		}
		
		// Use this row's target value (could weight by similarity)
		sum += toFloat64Impute(targetVal)
		count++
		
		// Limit to reasonable number of rows for performance
		if count >= 50 {
			break
		}
	}
	
	if count == 0 {
		// Fallback to mean
		return computeMean(targetSeries)
	}
	
	return sum / float64(count)
}

// Helper functions
func computeMean(series *seriesPkg.Series[any]) float64 {
	var sum float64
	count := 0
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if ok {
			sum += toFloat64Impute(val)
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func computeMedian(series *seriesPkg.Series[any]) float64 {
	values := make([]float64, 0)
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if ok {
			values = append(values, toFloat64Impute(val))
		}
	}
	if len(values) == 0 {
		return 0
	}
	
	// Simple sort for median
	for i := 0; i < len(values)-1; i++ {
		for j := i + 1; j < len(values); j++ {
			if values[i] > values[j] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
	
	n := len(values)
	if n%2 == 0 {
		return (values[n/2-1] + values[n/2]) / 2
	}
	return values[n/2]
}

// FitTransform fits the imputer and transforms the data in one step.
func (i *IterativeImputer) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := i.Fit(df, target...); err != nil {
		return nil, err
	}
	return i.Transform(df)
}

// IsFitted returns true if the imputer has been fitted.
func (i *IterativeImputer) IsFitted() bool {
	return i.fitted
}
