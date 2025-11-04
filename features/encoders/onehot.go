// Package encoders provides categorical encoding transformers.
package encoders

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// OneHotEncoder encodes categorical variables as binary vectors.
// Each category becomes a separate binary column.
type OneHotEncoder struct {
	// Columns to encode
	Columns []string
	
	// DropFirst drops the first category to avoid multicollinearity (dummy variable trap).
	// Default: false
	DropFirst bool
	
	// HandleUnknown specifies how to handle unknown categories during transform.
	// Options: "error" (default), "ignore"
	HandleUnknown string
	
	// Fitted categories for each column
	categories map[string][]string
	fitted     bool
}

// NewOneHotEncoder creates a new OneHotEncoder.
func NewOneHotEncoder(columns []string) *OneHotEncoder {
	return &OneHotEncoder{
		Columns:       columns,
		DropFirst:     false,
		HandleUnknown: "error",
		categories:    make(map[string][]string),
		fitted:        false,
	}
}

// Fit learns the unique categories for each column.
func (o *OneHotEncoder) Fit(df *dataframe.DataFrame, _ ...string) error {
	if len(o.Columns) == 0 {
		return fmt.Errorf("no columns specified for encoding")
	}
	
	o.categories = make(map[string][]string)
	
	for _, col := range o.Columns {
		colSeries, err := df.Column(col)
		if err != nil {
			return fmt.Errorf("column %q: %w", col, err)
		}
		
		// Get unique values
		unique := getUniqueStrings(colSeries)
		o.categories[col] = unique
	}
	
	o.fitted = true
	return nil
}

// Transform applies one-hot encoding to the data.
func (o *OneHotEncoder) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !o.fitted {
		return nil, fmt.Errorf("encoder not fitted")
	}
	
	result := df.Copy()
	
	for _, col := range o.Columns {
		colSeries, err := result.Column(col)
		if err != nil {
			return nil, fmt.Errorf("column %q: %w", col, err)
		}
		
		categories := o.categories[col]
		startIdx := 0
		if o.DropFirst {
			startIdx = 1
		}
		
		// Create binary column for each category
		for i := startIdx; i < len(categories); i++ {
			category := categories[i]
			newColName := fmt.Sprintf("%s_%s", col, category)
			
			// Create binary values
			binary := make([]any, colSeries.Len())
			for j := 0; j < colSeries.Len(); j++ {
				val, ok := colSeries.Get(j)
				if !ok {
					binary[j] = int64(0) // Treat nulls as 0
					continue
				}
				
				valStr := toString(val)
				if valStr == category {
					binary[j] = int64(1)
				} else {
					binary[j] = int64(0)
				}
			}
			
			result = result.WithColumn(newColName, seriesPkg.New(newColName, binary, core.DtypeInt64))
		}
		
		// Drop original column
		result = result.Drop(col)
	}
	
	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (o *OneHotEncoder) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := o.Fit(df, target...); err != nil {
		return nil, err
	}
	return o.Transform(df)
}

// IsFitted returns true if the encoder has been fitted.
func (o *OneHotEncoder) IsFitted() bool {
	return o.fitted
}

// GetCategories returns the learned categories for each column.
func (o *OneHotEncoder) GetCategories() map[string][]string {
	categories := make(map[string][]string, len(o.categories))
	for k, v := range o.categories {
		cats := make([]string, len(v))
		copy(cats, v)
		categories[k] = cats
	}
	return categories
}

// Helper functions

func getUniqueStrings(series interface{ Len() int; Get(int) (any, bool) }) []string {
	seen := make(map[string]bool)
	unique := make([]string, 0)
	
	for i := 0; i < series.Len(); i++ {
		val, ok := series.Get(i)
		if !ok {
			continue
		}
		
		strVal := toString(val)
		if !seen[strVal] {
			seen[strVal] = true
			unique = append(unique, strVal)
		}
	}
	
	return unique
}

func toString(val any) string {
	switch v := val.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}
