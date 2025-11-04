package encoders

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// OrdinalEncoder encodes categorical variables with ordinal relationships.
// Unlike LabelEncoder, the user specifies the order of categories.
type OrdinalEncoder struct {
	// Column to encode
	Column string
	
	// Categories in order (e.g., ["low", "medium", "high"])
	Categories []string
	
	// Mapping from category to integer
	mapping map[string]int64
	fitted  bool
}

// NewOrdinalEncoder creates a new OrdinalEncoder with specified category order.
func NewOrdinalEncoder(column string, categories []string) *OrdinalEncoder {
	return &OrdinalEncoder{
		Column:     column,
		Categories: categories,
		mapping:    make(map[string]int64),
		fitted:     false,
	}
}

// Fit builds the mapping from the specified category order.
func (o *OrdinalEncoder) Fit(df *dataframe.DataFrame, _ ...string) error {
	if o.Column == "" {
		return fmt.Errorf("no column specified for encoding")
	}
	
	if len(o.Categories) == 0 {
		return fmt.Errorf("no categories specified for ordinal encoding")
	}
	
	// Build mapping from user-specified order
	o.mapping = make(map[string]int64, len(o.Categories))
	for i, category := range o.Categories {
		o.mapping[category] = int64(i)
	}
	
	o.fitted = true
	return nil
}

// Transform applies the ordinal encoding to the data.
func (o *OrdinalEncoder) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !o.fitted {
		return nil, fmt.Errorf("encoder not fitted")
	}
	
	colSeries, err := df.Column(o.Column)
	if err != nil {
		return nil, fmt.Errorf("column %q: %w", o.Column, err)
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
		intVal, exists := o.mapping[strVal]
		if !exists {
			return nil, fmt.Errorf("unknown category %q in column %q (not in specified order)", strVal, o.Column)
		}
		encoded[i] = intVal
	}
	
	result := df.Copy()
	result = result.WithColumn(o.Column, seriesPkg.New(o.Column, encoded, core.DtypeInt64))
	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (o *OrdinalEncoder) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := o.Fit(df, target...); err != nil {
		return nil, err
	}
	return o.Transform(df)
}

// IsFitted returns true if the encoder has been fitted.
func (o *OrdinalEncoder) IsFitted() bool {
	return o.fitted
}

// GetMapping returns the category to integer mapping.
func (o *OrdinalEncoder) GetMapping() map[string]int64 {
	mapping := make(map[string]int64, len(o.mapping))
	for k, v := range o.mapping {
		mapping[k] = v
	}
	return mapping
}
