package encoders

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// FrequencyEncoder encodes categories by their frequency (count or percentage).
// More frequent categories get higher values.
type FrequencyEncoder struct {
	// Columns to encode
	Columns []string
	
	// Normalize converts counts to frequencies (0-1 range). Default: false
	Normalize bool
	
	// Mapping from column -> category -> frequency
	mapping map[string]map[string]float64
	fitted  bool
}

// NewFrequencyEncoder creates a new FrequencyEncoder.
func NewFrequencyEncoder(columns []string) *FrequencyEncoder {
	return &FrequencyEncoder{
		Columns:   columns,
		Normalize: false,
		mapping:   make(map[string]map[string]float64),
		fitted:    false,
	}
}

// Fit learns the frequency of each category.
func (f *FrequencyEncoder) Fit(df *dataframe.DataFrame, _ ...string) error {
	if len(f.Columns) == 0 {
		return fmt.Errorf("no columns specified for encoding")
	}
	
	f.mapping = make(map[string]map[string]float64)
	
	for _, col := range f.Columns {
		colSeries, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		// Count frequencies
		counts := make(map[string]int)
		totalCount := 0
		
		for i := 0; i < colSeries.Len(); i++ {
			val, ok := colSeries.Get(i)
			if !ok {
				continue
			}
			
			category := toString(val)
			counts[category]++
			totalCount++
		}
		
		// Convert to frequencies
		f.mapping[col] = make(map[string]float64, len(counts))
		for category, count := range counts {
			if f.Normalize {
				f.mapping[col][category] = float64(count) / float64(totalCount)
			} else {
				f.mapping[col][category] = float64(count)
			}
		}
	}
	
	f.fitted = true
	return nil
}

// Transform applies the frequency encoding to the data.
func (f *FrequencyEncoder) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !f.fitted {
		return nil, fmt.Errorf("encoder not fitted")
	}
	
	result := df.Copy()
	
	for _, col := range f.Columns {
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
			frequency, exists := f.mapping[col][category]
			if !exists {
				// Unknown category: use 0
				frequency = 0.0
			}
			encoded[i] = frequency
		}
		
		result = result.WithColumn(col, seriesPkg.New(col, encoded, core.DtypeFloat64))
	}
	
	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (f *FrequencyEncoder) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := f.Fit(df, target...); err != nil {
		return nil, err
	}
	return f.Transform(df)
}

// IsFitted returns true if the encoder has been fitted.
func (f *FrequencyEncoder) IsFitted() bool {
	return f.fitted
}

// GetMapping returns the category to frequency mapping.
func (f *FrequencyEncoder) GetMapping() map[string]map[string]float64 {
	mapping := make(map[string]map[string]float64, len(f.mapping))
	for col, catMap := range f.mapping {
		mapping[col] = make(map[string]float64, len(catMap))
		for cat, val := range catMap {
			mapping[col][cat] = val
		}
	}
	return mapping
}
