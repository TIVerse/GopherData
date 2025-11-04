package scalers

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// MinMaxScaler scales features to a given range (default: [0, 1]).
// The transformation is given by: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
type MinMaxScaler struct {
	// Columns to scale. If nil, all numeric columns are scaled.
	Columns []string
	
	// FeatureRange defines the desired range of transformed data.
	// Default: [0, 1]
	FeatureMin float64
	FeatureMax float64
	
	// Fitted statistics
	mins   map[string]float64
	maxs   map[string]float64
	fitted bool
}

// NewMinMaxScaler creates a new MinMaxScaler with default range [0, 1].
func NewMinMaxScaler(columns []string) *MinMaxScaler {
	return &MinMaxScaler{
		Columns:    columns,
		FeatureMin: 0.0,
		FeatureMax: 1.0,
		mins:       make(map[string]float64),
		maxs:       make(map[string]float64),
		fitted:     false,
	}
}

// Fit computes the min and max for each column.
func (m *MinMaxScaler) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := m.Columns
	if cols == nil {
		cols = getNumericColumns(df)
	}
	
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to scale")
	}
	
	m.mins = make(map[string]float64)
	m.maxs = make(map[string]float64)
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		min, max := computeMinMax(series)
		m.mins[col] = min
		m.maxs[col] = max
	}
	
	m.fitted = true
	return nil
}

// Transform applies the min-max scaling to the data.
func (m *MinMaxScaler) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !m.fitted {
		return nil, fmt.Errorf("scaler not fitted")
	}
	
	result := df.Copy()
	
	for col := range m.mins {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		min := m.mins[col]
		max := m.maxs[col]
		dataRange := max - min
		
		// Skip columns with zero range
		if dataRange == 0 {
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
			// Scale to [0, 1] then to [FeatureMin, FeatureMax]
			stdVal := (floatVal - min) / dataRange
			scaled[i] = stdVal*(m.FeatureMax-m.FeatureMin) + m.FeatureMin
		}
		
		result = result.WithColumn(col, seriesPkg.New(col, scaled, core.DtypeFloat64))
	}
	
	return result, nil
}

// FitTransform fits the scaler and transforms the data in one step.
func (m *MinMaxScaler) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := m.Fit(df, target...); err != nil {
		return nil, err
	}
	return m.Transform(df)
}

// IsFitted returns true if the scaler has been fitted.
func (m *MinMaxScaler) IsFitted() bool {
	return m.fitted
}

// GetMins returns the computed minimums.
func (m *MinMaxScaler) GetMins() map[string]float64 {
	mins := make(map[string]float64, len(m.mins))
	for k, v := range m.mins {
		mins[k] = v
	}
	return mins
}

// GetMaxs returns the computed maximums.
func (m *MinMaxScaler) GetMaxs() map[string]float64 {
	maxs := make(map[string]float64, len(m.maxs))
	for k, v := range m.maxs {
		maxs[k] = v
	}
	return maxs
}

func computeMinMax(series interface{ Len() int; Get(int) (any, bool) }) (float64, float64) {
	min := math.Inf(1)
	max := math.Inf(-1)
	found := false
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		floatVal := toFloat64(val)
		if !found {
			min = floatVal
			max = floatVal
			found = true
		} else {
			if floatVal < min {
				min = floatVal
			}
			if floatVal > max {
				max = floatVal
			}
		}
	}
	
	if !found {
		return 0, 0
	}
	return min, max
}
