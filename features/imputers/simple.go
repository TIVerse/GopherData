// Package imputers provides missing value imputation transformers.
package imputers

import (
	"fmt"
	"sort"

	"github.com/TIVerse/GopherData/dataframe"
)

// SimpleImputer fills missing values using basic strategies.
type SimpleImputer struct {
	// Columns to impute. If nil, all columns with nulls are imputed.
	Columns []string
	
	// Strategy for imputation:
	// - "mean": replace with mean (numeric only)
	// - "median": replace with median (numeric only)
	// - "most_frequent": replace with mode
	// - "constant": replace with FillValue
	Strategy string
	
	// FillValue is used when Strategy="constant"
	FillValue any
	
	// Fitted statistics for each column
	stats  map[string]any
	fitted bool
}

// NewSimpleImputer creates a new SimpleImputer with the given strategy.
func NewSimpleImputer(columns []string, strategy string) *SimpleImputer {
	return &SimpleImputer{
		Columns:  columns,
		Strategy: strategy,
		stats:    make(map[string]any),
		fitted:   false,
	}
}

// Fit computes the statistics for imputation.
func (s *SimpleImputer) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := s.Columns
	if cols == nil {
		cols = df.Columns()
	}
	
	if len(cols) == 0 {
		return fmt.Errorf("no columns to impute")
	}
	
	s.stats = make(map[string]any)
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		// Skip columns with no nulls
		if !hasNulls(series) {
			continue
		}
		
		switch s.Strategy {
		case "mean":
			s.stats[col] = computeSeriesMean(series)
		case "median":
			s.stats[col] = computeSeriesMedian(series)
		case "most_frequent":
			s.stats[col] = computeMode(series)
		case "constant":
			s.stats[col] = s.FillValue
		default:
			return fmt.Errorf("unknown strategy: %s", s.Strategy)
		}
	}
	
	s.fitted = true
	return nil
}

// Transform applies the imputation to the data.
func (s *SimpleImputer) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !s.fitted {
		return nil, fmt.Errorf("imputer not fitted")
	}
	
	result := df.Copy()
	
	for col, fillValue := range s.stats {
		result = result.FillNAColumn(col, fillValue)
	}
	
	return result, nil
}

// FitTransform fits the imputer and transforms the data in one step.
func (s *SimpleImputer) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := s.Fit(df, target...); err != nil {
		return nil, err
	}
	return s.Transform(df)
}

// IsFitted returns true if the imputer has been fitted.
func (s *SimpleImputer) IsFitted() bool {
	return s.fitted
}

// GetStats returns the computed statistics for each column.
func (s *SimpleImputer) GetStats() map[string]any {
	stats := make(map[string]any, len(s.stats))
	for k, v := range s.stats {
		stats[k] = v
	}
	return stats
}

// Helper functions

func hasNulls(series interface{ Len() int; Get(int) (any, bool) }) bool {
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok || val == nil {
			return true
		}
	}
	return false
}

func computeSeriesMean(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	var sum float64
	count := 0
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok || val == nil {
			continue
		}
		sum += toFloat64Impute(val)
		count++
	}
	
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func computeSeriesMedian(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	values := make([]float64, 0)
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok || val == nil {
			continue
		}
		values = append(values, toFloat64Impute(val))
	}
	
	if len(values) == 0 {
		return 0
	}
	
	sort.Float64s(values)
	n := len(values)
	if n%2 == 0 {
		return (values[n/2-1] + values[n/2]) / 2
	}
	return values[n/2]
}

func computeMode(series interface{ Len() int; Get(int) (any, bool) }) any {
	counts := make(map[string]int)
	values := make(map[string]any)
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		key := fmt.Sprintf("%v", val)
		counts[key]++
		values[key] = val
	}
	
	if len(counts) == 0 {
		return nil
	}
	
	// Find most frequent
	maxCount := 0
	var mode string
	for key, count := range counts {
		if count > maxCount {
			maxCount = count
			mode = key
		}
	}
	
	return values[mode]
}

func toFloat64Impute(val any) float64 {
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
