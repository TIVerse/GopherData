package scalers

import (
	"fmt"
	"math"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// MaxAbsScaler scales each feature by its maximum absolute value.
// This scaler is particularly suited for sparse data as it preserves zero values.
// The transformation is: X_scaled = X / max(abs(X))
type MaxAbsScaler struct {
	// Columns to scale. If nil, all numeric columns are scaled.
	Columns []string
	
	// Fitted statistics
	maxAbss map[string]float64
	fitted  bool
}

// NewMaxAbsScaler creates a new MaxAbsScaler.
func NewMaxAbsScaler(columns []string) *MaxAbsScaler {
	return &MaxAbsScaler{
		Columns: columns,
		maxAbss: make(map[string]float64),
		fitted:  false,
	}
}

// Fit computes the maximum absolute value for each column.
func (m *MaxAbsScaler) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := m.Columns
	if cols == nil {
		cols = getNumericColumns(df)
	}
	
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to scale")
	}
	
	m.maxAbss = make(map[string]float64)
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		m.maxAbss[col] = computeMaxAbs(series)
	}
	
	m.fitted = true
	return nil
}

// Transform applies the max-abs scaling to the data.
func (m *MaxAbsScaler) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !m.fitted {
		return nil, fmt.Errorf("scaler not fitted")
	}
	
	result := df.Copy()
	
	for col, maxAbs := range m.maxAbss {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		// Skip columns with zero max absolute value
		if maxAbs == 0 {
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
			scaled[i] = floatVal / maxAbs
		}
		
		result = result.WithColumn(col, seriesPkg.New(col, scaled, core.DtypeFloat64))
	}
	
	return result, nil
}

// FitTransform fits the scaler and transforms the data in one step.
func (m *MaxAbsScaler) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := m.Fit(df, target...); err != nil {
		return nil, err
	}
	return m.Transform(df)
}

// IsFitted returns true if the scaler has been fitted.
func (m *MaxAbsScaler) IsFitted() bool {
	return m.fitted
}

// GetMaxAbss returns the computed maximum absolute values.
func (m *MaxAbsScaler) GetMaxAbss() map[string]float64 {
	maxAbss := make(map[string]float64, len(m.maxAbss))
	for k, v := range m.maxAbss {
		maxAbss[k] = v
	}
	return maxAbss
}

func computeMaxAbs(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	maxAbs := 0.0
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		absVal := math.Abs(toFloat64(val))
		if absVal > maxAbs {
			maxAbs = absVal
		}
	}
	
	return maxAbs
}
