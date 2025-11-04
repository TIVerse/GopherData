// Package selectors provides feature selection transformers.
package selectors

import (
	"fmt"

	"github.com/TIVerse/GopherData/dataframe"
)

// VarianceThreshold removes features with variance below a threshold.
// Features with low variance are typically not useful for prediction.
type VarianceThreshold struct {
	// Threshold is the minimum variance required
	Threshold float64
	
	// Selected column names
	selected []string
	fitted   bool
}

// NewVarianceThreshold creates a new VarianceThreshold selector.
func NewVarianceThreshold(threshold float64) *VarianceThreshold {
	return &VarianceThreshold{
		Threshold: threshold,
		selected:  nil,
		fitted:    false,
	}
}

// Fit computes the variance for each numeric column and selects features.
func (v *VarianceThreshold) Fit(df *dataframe.DataFrame, _ ...string) error {
	v.selected = make([]string, 0)
	
	cols := getNumericColumns(df)
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to select from")
	}
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			continue
		}
		
		variance := computeVariance(series)
		if variance > v.Threshold {
			v.selected = append(v.selected, col)
		}
	}
	
	v.fitted = true
	return nil
}

// Transform returns a DataFrame with only the selected features.
func (v *VarianceThreshold) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !v.fitted {
		return nil, fmt.Errorf("selector not fitted")
	}
	
	return df.Select(v.selected...), nil
}

// FitTransform fits the selector and transforms the data in one step.
func (v *VarianceThreshold) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := v.Fit(df, target...); err != nil {
		return nil, err
	}
	return v.Transform(df)
}

// IsFitted returns true if the selector has been fitted.
func (v *VarianceThreshold) IsFitted() bool {
	return v.fitted
}

// GetSelectedFeatures returns the list of selected feature names.
func (v *VarianceThreshold) GetSelectedFeatures() []string {
	selected := make([]string, len(v.selected))
	copy(selected, v.selected)
	return selected
}

// Helper functions

func getNumericColumns(df *dataframe.DataFrame) []string {
	numericCols := make([]string, 0)
	cols := df.Columns()
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			continue
		}
		
		// Check if column is numeric
		if isNumericSeries(series) {
			numericCols = append(numericCols, col)
		}
	}
	
	return numericCols
}

func isNumericSeries(series interface{ Len() int; Get(int) (any, bool) }) bool {
	for i := 0; i < series.Len() && i < 10; i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		switch val.(type) {
		case float64, float32, int, int64, int32, int16, int8:
			return true
		default:
			return false
		}
	}
	return false
}

func computeVariance(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	// Compute mean
	var sum float64
	count := 0
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		sum += toFloat64Selector(val)
		count++
	}
	
	if count < 2 {
		return 0
	}
	
	mean := sum / float64(count)
	
	// Compute variance
	var sumSq float64
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		diff := toFloat64Selector(val) - mean
		sumSq += diff * diff
	}
	
	return sumSq / float64(count-1)
}

func toFloat64Selector(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	default:
		return 0
	}
}
