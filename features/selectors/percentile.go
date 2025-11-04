package selectors

import (
	"fmt"

	"github.com/TIVerse/GopherData/dataframe"
)

// SelectPercentile selects features based on a percentile of the highest scores.
type SelectPercentile struct {
	// Percentile (0-100) of features to keep
	Percentile float64
	
	// Selected column names
	selected []string
	scores   map[string]float64
	fitted   bool
}

// NewSelectPercentile creates a new SelectPercentile selector.
func NewSelectPercentile(percentile float64) *SelectPercentile {
	return &SelectPercentile{
		Percentile: percentile,
		selected:   nil,
		scores:     make(map[string]float64),
		fitted:     false,
	}
}

// Fit computes scores and selects features above the percentile threshold.
func (s *SelectPercentile) Fit(df *dataframe.DataFrame, target ...string) error {
	if len(target) == 0 {
		return fmt.Errorf("target column required for SelectPercentile")
	}
	
	// Use SelectKBest logic but select by percentile
	cols := getNumericColumns(df)
	k := int(float64(len(cols)) * s.Percentile / 100.0)
	if k < 1 {
		k = 1
	}
	
	kbest := NewSelectKBest(k)
	if err := kbest.Fit(df, target...); err != nil {
		return err
	}
	
	s.selected = kbest.GetSelectedFeatures()
	s.scores = kbest.GetScores()
	s.fitted = true
	return nil
}

// Transform returns a DataFrame with selected features.
func (s *SelectPercentile) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	if !s.fitted {
		return nil, fmt.Errorf("selector not fitted")
	}
	
	return df.Select(s.selected...), nil
}

// FitTransform fits the selector and transforms the data in one step.
func (s *SelectPercentile) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := s.Fit(df, target...); err != nil {
		return nil, err
	}
	return s.Transform(df)
}

// IsFitted returns true if the selector has been fitted.
func (s *SelectPercentile) IsFitted() bool {
	return s.fitted
}

// GetSelectedFeatures returns the list of selected feature names.
func (s *SelectPercentile) GetSelectedFeatures() []string {
	selected := make([]string, len(s.selected))
	copy(selected, s.selected)
	return selected
}
