package encoders

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// LabelEncoder encodes categorical variables as integers.
// Each unique category is assigned an integer from 0 to n_categories-1.
type LabelEncoder struct {
	// Column to encode
	Column string
	
	// Mapping from category to integer
	mapping        map[string]int64
	reverseMapping map[int64]string
	fitted         bool
}

// NewLabelEncoder creates a new LabelEncoder.
func NewLabelEncoder(column string) *LabelEncoder {
	return &LabelEncoder{
		Column:         column,
		mapping:        make(map[string]int64),
		reverseMapping: make(map[int64]string),
		fitted:         false,
	}
}

// Fit learns the mapping from categories to integers.
func (l *LabelEncoder) Fit(df *dataframe.DataFrame, _ ...string) error {
	if l.Column == "" {
		return fmt.Errorf("no column specified for encoding")
	}
	
	colSeries, err := df.Column(l.Column)
	if err != nil {
		return fmt.Errorf("column %q: %w", l.Column, err)
	}
	
	// Get unique values and assign integers
	unique := getUniqueStrings(colSeries)
	l.mapping = make(map[string]int64, len(unique))
	l.reverseMapping = make(map[int64]string, len(unique))
	
	for i, category := range unique {
		l.mapping[category] = int64(i)
		l.reverseMapping[int64(i)] = category
	}
	
	l.fitted = true
	return nil
}

// Transform applies the label encoding to the data.
func (l *LabelEncoder) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !l.fitted {
		return nil, fmt.Errorf("encoder not fitted")
	}
	
	colSeries, err := df.Column(l.Column)
	if err != nil {
		return nil, fmt.Errorf("column %q: %w", l.Column, err)
	}
	
	// Encode values
	encoded := make([]any, colSeries.Len())
	for i := 0; i < colSeries.Len(); i++ {
		val, ok := colSeries.Get(i)
		if !ok {
			encoded[i] = nil
			continue
		}
		
		strVal := toString(val)
		intVal, exists := l.mapping[strVal]
		if !exists {
			return nil, fmt.Errorf("unknown category %q in column %q", strVal, l.Column)
		}
		encoded[i] = intVal
	}
	
	result := df.Copy()
	result = result.WithColumn(l.Column, seriesPkg.New(l.Column, encoded, core.DtypeInt64))
	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (l *LabelEncoder) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := l.Fit(df, target...); err != nil {
		return nil, err
	}
	return l.Transform(df)
}

// IsFitted returns true if the encoder has been fitted.
func (l *LabelEncoder) IsFitted() bool {
	return l.fitted
}

// GetMapping returns the category to integer mapping.
func (l *LabelEncoder) GetMapping() map[string]int64 {
	mapping := make(map[string]int64, len(l.mapping))
	for k, v := range l.mapping {
		mapping[k] = v
	}
	return mapping
}

// InverseTransform converts encoded integers back to categories.
func (l *LabelEncoder) InverseTransform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !l.fitted {
		return nil, fmt.Errorf("encoder not fitted")
	}
	
	colSeries, err := df.Column(l.Column)
	if err != nil {
		return nil, fmt.Errorf("column %q: %w", l.Column, err)
	}
	
	// Decode values
	decoded := make([]any, colSeries.Len())
	for i := 0; i < colSeries.Len(); i++ {
		val, ok := colSeries.Get(i)
		if !ok {
			decoded[i] = nil
			continue
		}
		
		intVal := toInt64(val)
		category, exists := l.reverseMapping[intVal]
		if !exists {
			return nil, fmt.Errorf("unknown encoded value %d in column %q", intVal, l.Column)
		}
		decoded[i] = category
	}
	
	result := df.Copy()
	result = result.WithColumn(l.Column, seriesPkg.New(l.Column, decoded, core.DtypeString))
	return result, nil
}

func toInt64(val any) int64 {
	switch v := val.(type) {
	case int64:
		return v
	case int:
		return int64(v)
	case int32:
		return int64(v)
	case int16:
		return int64(v)
	case int8:
		return int64(v)
	default:
		return 0
	}
}
