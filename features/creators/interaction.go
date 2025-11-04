package creators

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// InteractionFeatures creates pairwise interaction features.
// For columns [a, b, c]: creates [a×b, a×c, b×c]
type InteractionFeatures struct {
	// Columns to create interactions from. If nil, use all numeric columns.
	Columns []string
	
	fitted bool
}

// NewInteractionFeatures creates a new InteractionFeatures transformer.
func NewInteractionFeatures(columns []string) *InteractionFeatures {
	return &InteractionFeatures{
		Columns: columns,
		fitted:  false,
	}
}

// Fit is a no-op for InteractionFeatures (stateless transformer).
func (i *InteractionFeatures) Fit(df *dataframe.DataFrame, _ ...string) error {
	cols := i.Columns
	if cols == nil {
		cols = getNumericColumns(df)
	}
	
	if len(cols) < 2 {
		return fmt.Errorf("need at least 2 columns to create interactions")
	}
	
	i.Columns = cols
	i.fitted = true
	return nil
}

// Transform creates pairwise interaction features.
func (i *InteractionFeatures) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !i.fitted {
		return nil, fmt.Errorf("transformer not fitted")
	}
	
	result := df.Copy()
	
	// Create all pairwise interactions
	for idx1, col1 := range i.Columns {
		for idx2 := idx1 + 1; idx2 < len(i.Columns); idx2++ {
			col2 := i.Columns[idx2]
			
			colSeries1, err := df.Column(col1)
			if err != nil {
				continue
			}
			
			colSeries2, err := df.Column(col2)
			if err != nil {
				continue
			}
			
			// Create interaction feature
			interaction := make([]any, colSeries1.Len())
			for j := 0; j < colSeries1.Len(); j++ {
				val1, ok1 := colSeries1.Get(j)
				val2, ok2 := colSeries2.Get(j)
				
				if !ok1 || !ok2 {
					interaction[j] = nil
					continue
				}
				
				interaction[j] = toFloat64Creator(val1) * toFloat64Creator(val2)
			}
			
			newColName := fmt.Sprintf("%s*%s", col1, col2)
			result = result.WithColumn(newColName, seriesPkg.New(newColName, interaction, core.DtypeFloat64))
		}
	}
	
	return result, nil
}

// FitTransform fits the transformer and transforms the data in one step.
func (i *InteractionFeatures) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := i.Fit(df, target...); err != nil {
		return nil, err
	}
	return i.Transform(df)
}

// IsFitted returns true if the transformer has been fitted.
func (i *InteractionFeatures) IsFitted() bool {
	return i.fitted
}
