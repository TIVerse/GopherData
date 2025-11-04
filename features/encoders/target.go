package encoders

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// TargetEncoder encodes categories using the mean of the target variable.
// This is useful for high-cardinality categorical features.
// Uses smoothing to prevent overfitting: (count * mean + smooth * global_mean) / (count + smooth)
type TargetEncoder struct {
	// Columns to encode
	Columns []string
	
	// Target column name (provided during Fit)
	Target string
	
	// Smooth is the smoothing factor. Higher values give more weight to global mean.
	// Default: 1.0
	Smooth float64
	
	// Mapping from column -> category -> encoded value
	mapping    map[string]map[string]float64
	globalMean float64
	fitted     bool
}

// NewTargetEncoder creates a new TargetEncoder.
func NewTargetEncoder(columns []string) *TargetEncoder {
	return &TargetEncoder{
		Columns: columns,
		Smooth:  1.0,
		mapping: make(map[string]map[string]float64),
		fitted:  false,
	}
}

// Fit learns the target mean for each category.
func (t *TargetEncoder) Fit(df *dataframe.DataFrame, target ...string) error {
	if len(target) == 0 {
		return fmt.Errorf("target column required for TargetEncoder")
	}
	t.Target = target[0]
	
	if len(t.Columns) == 0 {
		return fmt.Errorf("no columns specified for encoding")
	}
	
	// Get target series
	targetSeries, err := df.Column(t.Target)
	if err != nil {
		return fmt.Errorf("target column %q: %w", t.Target, err)
	}
	
	// Compute global mean
	t.globalMean = computeTargetMean(targetSeries)
	
	// Compute mean for each category in each column
	t.mapping = make(map[string]map[string]float64)
	
	for _, col := range t.Columns {
		colSeries, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		// Group by category and compute smoothed mean
		categoryStats := make(map[string]*categoryTargetStats)
		
		for i := 0; i < colSeries.Len(); i++ {
			catVal, ok := colSeries.Get(i)
			if !ok {
				continue
			}
			
			targetVal, ok := targetSeries.Get(i)
			if !ok {
				continue
			}
			
			category := toString(catVal)
			if categoryStats[category] == nil {
				categoryStats[category] = &categoryTargetStats{}
			}
			
			categoryStats[category].sum += toFloat64Target(targetVal)
			categoryStats[category].count++
		}
		
		// Apply smoothing and store mapping
		t.mapping[col] = make(map[string]float64)
		for category, stats := range categoryStats {
			if stats.count == 0 {
				t.mapping[col][category] = t.globalMean
			} else {
				// Smoothed mean: (count * mean + smooth * global_mean) / (count + smooth)
				smoothedMean := (stats.sum + t.Smooth*t.globalMean) / (float64(stats.count) + t.Smooth)
				t.mapping[col][category] = smoothedMean
			}
		}
	}
	
	t.fitted = true
	return nil
}

// Transform applies the target encoding to the data.
func (t *TargetEncoder) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !t.fitted {
		return nil, fmt.Errorf("encoder not fitted")
	}
	
	result := df.Copy()
	
	for _, col := range t.Columns {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		// Encode values
		encoded := make([]any, colSeries.Len())
		for i := 0; i < colSeries.Len(); i++ {
			val, ok := colSeries.Get(i)
			if !ok {
				encoded[i] = nil
				continue
			}
			
			category := toString(val)
			encodedVal, exists := t.mapping[col][category]
			if !exists {
				// Unknown category: use global mean
				encodedVal = t.globalMean
			}
			encoded[i] = encodedVal
		}
		
		result = result.WithColumn(col, seriesPkg.New(col, encoded, core.DtypeFloat64))
	}
	
	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (t *TargetEncoder) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := t.Fit(df, target...); err != nil {
		return nil, err
	}
	return t.Transform(df)
}

// IsFitted returns true if the encoder has been fitted.
func (t *TargetEncoder) IsFitted() bool {
	return t.fitted
}

// GetMapping returns the category to encoded value mapping.
func (t *TargetEncoder) GetMapping() map[string]map[string]float64 {
	mapping := make(map[string]map[string]float64, len(t.mapping))
	for col, catMap := range t.mapping {
		mapping[col] = make(map[string]float64, len(catMap))
		for cat, val := range catMap {
			mapping[col][cat] = val
		}
	}
	return mapping
}

type categoryTargetStats struct {
	sum   float64
	count int
}

func computeTargetMean(series interface{ Len() int; Get(int) (any, bool) }) float64 {
	var sum float64
	count := 0
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		sum += toFloat64Target(val)
		count++
	}
	
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func toFloat64Target(val any) float64 {
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
	case bool:
		if v {
			return 1.0
		}
		return 0.0
	default:
		return 0
	}
}
