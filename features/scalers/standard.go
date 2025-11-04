// Package scalers provides data normalization and standardization transformers.
package scalers

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// StandardScaler standardizes features by removing the mean and scaling to unit variance.
// The standard score of a sample x is calculated as: z = (x - μ) / σ
// where μ is the mean and σ is the standard deviation.
type StandardScaler struct {
	// Columns to scale. If nil, all numeric columns are scaled.
	Columns []string
	
	// WithMean centers the data before scaling. Default: true
	WithMean bool
	
	// WithStd scales the data to unit variance. Default: true
	WithStd bool
	
	// Fitted statistics
	means  map[string]float64
	stds   map[string]float64
	fitted bool
}

// NewStandardScaler creates a new StandardScaler with default settings.
func NewStandardScaler(columns []string) *StandardScaler {
	return &StandardScaler{
		Columns:  columns,
		WithMean: true,
		WithStd:  true,
		means:    make(map[string]float64),
		stds:     make(map[string]float64),
		fitted:   false,
	}
}

// Fit computes the mean and standard deviation for each column.
func (s *StandardScaler) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := s.Columns
	if cols == nil {
		// Get all numeric columns
		cols = getNumericColumns(df)
	}
	
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to scale")
	}
	
	s.means = make(map[string]float64)
	s.stds = make(map[string]float64)
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		// Compute mean
		var mean float64
		if s.WithMean {
			mean = computeMean(series)
			s.means[col] = mean
		}
		
		// Compute standard deviation
		var std float64
		if s.WithStd {
			std = computeStd(series, mean)
			s.stds[col] = std
		}
	}
	
	s.fitted = true
	return nil
}

// Transform applies the standardization to the data.
func (s *StandardScaler) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !s.fitted {
		return nil, fmt.Errorf("scaler not fitted")
	}
	
	result := df.Copy()
	
	for col := range s.means {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		mean := s.means[col]
		std := s.stds[col]
		
		// Skip columns with zero variance
		if s.WithStd && std == 0 {
			continue
		}
		
		// Apply transformation
		scaled := make([]any, colSeries.Len())
		for i := 0; i < colSeries.Len(); i++ {
			val, ok := colSeries.Get(i)
			if !ok {
				scaled[i] = nil
				continue
			}
			
			floatVal := toFloat64(val)
			transformed := floatVal
			
			if s.WithMean {
				transformed -= mean
			}
			if s.WithStd && std != 0 {
				transformed /= std
			}
			
			scaled[i] = transformed
		}
		
		result = result.WithColumn(col, createFloatSeries(col, scaled))
	}
	
	return result, nil
}

// FitTransform fits the scaler and transforms the data in one step.
func (s *StandardScaler) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := s.Fit(df, target...); err != nil {
		return nil, err
	}
	return s.Transform(df)
}

// IsFitted returns true if the scaler has been fitted.
func (s *StandardScaler) IsFitted() bool {
	return s.fitted
}

// GetMeans returns the computed means.
func (s *StandardScaler) GetMeans() map[string]float64 {
	means := make(map[string]float64, len(s.means))
	for k, v := range s.means {
		means[k] = v
	}
	return means
}

// GetStds returns the computed standard deviations.
func (s *StandardScaler) GetStds() map[string]float64 {
	stds := make(map[string]float64, len(s.stds))
	for k, v := range s.stds {
		stds[k] = v
	}
	return stds
}

// Helper functions

func computeMean(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	var sum float64
	count := 0
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok || val == nil {
			continue
		}
		sum += toFloat64(val)
		count++
	}
	
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func computeStd(series interface{ Len() int; Get(int) (any, bool) }, mean float64) float64 {
	var sumSq float64
	count := 0
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok || val == nil {
			continue
		}
		diff := toFloat64(val) - mean
		sumSq += diff * diff
		count++
	}
	
	if count < 2 {
		return 0
	}
	
	return math.Sqrt(sumSq / float64(count-1))
}

func toFloat64(val any) float64 {
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
	case int16:
		return float64(v)
	case int8:
		return float64(v)
	default:
		return 0
	}
}

func getNumericColumns(df *dataframe.DataFrame) []string {
	numericCols := make([]string, 0)
	cols := df.Columns()
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			continue
		}
		
		// Check if column is numeric by examining first non-null value
		for i := 0; i < series.Len(); i++ {
			val, ok := series.Get(i)
			if !ok {
				continue
			}
			
			switch val.(type) {
			case float64, float32, int, int64, int32, int16, int8:
				numericCols = append(numericCols, col)
			}
			break
		}
	}
	
	return numericCols
}

func createFloatSeries(name string, data []any) *seriesPkg.Series[any] {
	return seriesPkg.New(name, data, core.DtypeFloat64)
}
