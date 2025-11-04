package scalers

import (
	"fmt"
	"sort"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// RobustScaler scales features using statistics that are robust to outliers.
// It removes the median and scales the data according to the Interquartile Range (IQR).
// The transformation is: X_scaled = (X - median) / IQR
type RobustScaler struct {
	// Columns to scale. If nil, all numeric columns are scaled.
	Columns []string
	
	// WithCentering centers the data before scaling. Default: true
	WithCentering bool
	
	// WithScaling scales the data to IQR. Default: true
	WithScaling bool
	
	// QuantileRange for IQR calculation. Default: [25.0, 75.0]
	QuantileRange [2]float64
	
	// Fitted statistics
	medians map[string]float64
	iqrs    map[string]float64
	fitted  bool
}

// NewRobustScaler creates a new RobustScaler with default settings.
func NewRobustScaler(columns []string) *RobustScaler {
	return &RobustScaler{
		Columns:       columns,
		WithCentering: true,
		WithScaling:   true,
		QuantileRange: [2]float64{25.0, 75.0},
		medians:       make(map[string]float64),
		iqrs:          make(map[string]float64),
		fitted:        false,
	}
}

// Fit computes the median and IQR for each column.
func (r *RobustScaler) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := r.Columns
	if cols == nil {
		cols = getNumericColumns(df)
	}
	
	if len(cols) == 0 {
		return fmt.Errorf("no numeric columns to scale")
	}
	
	r.medians = make(map[string]float64)
	r.iqrs = make(map[string]float64)
	
	for _, col := range cols {
		series, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		// Compute median
		if r.WithCentering {
			r.medians[col] = computeMedian(series)
		}
		
		// Compute IQR
		if r.WithScaling {
			q1 := computeQuantile(series, r.QuantileRange[0]/100.0)
			q3 := computeQuantile(series, r.QuantileRange[1]/100.0)
			r.iqrs[col] = q3 - q1
		}
	}
	
	r.fitted = true
	return nil
}

// Transform applies the robust scaling to the data.
func (r *RobustScaler) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !r.fitted {
		return nil, fmt.Errorf("scaler not fitted")
	}
	
	result := df.Copy()
	
	for col := range r.medians {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		median := r.medians[col]
		iqr := r.iqrs[col]
		
		// Skip columns with zero IQR
		if r.WithScaling && iqr == 0 {
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
			
			if r.WithCentering {
				transformed -= median
			}
			if r.WithScaling && iqr != 0 {
				transformed /= iqr
			}
			
			scaled[i] = transformed
		}
		
		result = result.WithColumn(col, seriesPkg.New(col, scaled, core.DtypeFloat64))
	}
	
	return result, nil
}

// FitTransform fits the scaler and transforms the data in one step.
func (r *RobustScaler) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := r.Fit(df, target...); err != nil {
		return nil, err
	}
	return r.Transform(df)
}

// IsFitted returns true if the scaler has been fitted.
func (r *RobustScaler) IsFitted() bool {
	return r.fitted
}

// GetMedians returns the computed medians.
func (r *RobustScaler) GetMedians() map[string]float64 {
	medians := make(map[string]float64, len(r.medians))
	for k, v := range r.medians {
		medians[k] = v
	}
	return medians
}

// GetIQRs returns the computed IQRs.
func (r *RobustScaler) GetIQRs() map[string]float64 {
	iqrs := make(map[string]float64, len(r.iqrs))
	for k, v := range r.iqrs {
		iqrs[k] = v
	}
	return iqrs
}

func computeMedian(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	return computeQuantile(series, 0.5)
}

func computeQuantile(series interface{ Len() int; Get(int) (any, bool) }, q float64) float64 {
	// Collect non-null values
	values := make([]float64, 0, series.Len())
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		values = append(values, toFloat64(val))
	}
	
	if len(values) == 0 {
		return 0
	}
	
	// Sort values
	sort.Float64s(values)
	
	// Compute quantile position
	pos := q * float64(len(values)-1)
	lower := int(pos)
	upper := lower + 1
	
	if upper >= len(values) {
		return values[len(values)-1]
	}
	
	// Linear interpolation
	fraction := pos - float64(lower)
	return values[lower]*(1-fraction) + values[upper]*fraction
}
